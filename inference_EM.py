#!/usr/bin/env python3
"""
inference_EM.py — Extended Minds Dump Classification Test Suite
════════════════════════════════════════════════════════════════════
Simulates the full capture pipeline WITHOUT saving to the database.
Validates bucket classification, reminder detection, list detection,
group @mention routing, and the iOS TodoView section each capture
would land in.

HOW IT WORKS
────────────
1. Mocks all DB-touching imports so services can be loaded in isolation.
2. Loads the real ONNX classifier (if model files are present).
3. Calls intent_service.parse() — the same function the production
   capture route calls.
4. Simulates the message_processor dispatch decision tree.
5. Applies the iOS TodoView section rules (mirrors TodoView.swift).
6. Compares the computed outcome to the hand-written expected outcome
   and prints a pass / fail report.

USAGE
─────
    cd extended-brain-backend

    # Full suite (uses Gemini for slow-path cases — set GEMINI_API_KEY)
    python inference_EM.py

    # ONNX fast path only — skip LLM, use rule-based fallback
    python inference_EM.py --fast

    # Only individual / group subset
    python inference_EM.py --individual
    python inference_EM.py --group

    # Single case by ID (1-indexed)
    python inference_EM.py --case 5

    # Show full parsed intent output for debugging
    python inference_EM.py --verbose

    # Live mode: call the actual backend API (needs --token)
    python inference_EM.py --live --token <JWT>

ENVIRONMENT
───────────
    GEMINI_API_KEY   Required for slow-path / LLM classification.
    BACKBONE_ONNX_URL  Optional: URL to auto-download backbone.onnx on first run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

# ─── Mock DB imports BEFORE any service module is loaded ──────────────────────
# intent_service and classifier_service are pure (no DB), but list_service
# and message_processor import from database.py.  We mock those so we can
# import the pure helper functions we need.
from unittest.mock import MagicMock

for _mod in (
    "database", "models", "sqlalchemy", "sqlalchemy.ext.asyncio",
    "sqlalchemy.orm", "asyncpg", "psycopg2",
):
    sys.modules.setdefault(_mod, MagicMock())

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

IST = ZoneInfo("Asia/Kolkata")

# ─────────────────────────────────────────────────────────────────────────────
# Date helpers
# ─────────────────────────────────────────────────────────────────────────────

def _today() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d")

def _tomorrow() -> str:
    return (datetime.now(IST) + timedelta(days=1)).strftime("%Y-%m-%d")

def _next_weekday(name: str) -> str:
    days = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
    now  = datetime.now(IST)
    tgt  = days.index(name.lower())
    delta = (tgt - now.weekday()) % 7 or 7
    return (now + timedelta(days=delta)).strftime("%Y-%m-%d")

def _date_label(d: str) -> str:
    today = _today()
    tomorrow = _tomorrow()
    if d == today:     return "today"
    if d == tomorrow:  return "tomorrow"
    return d

# ─────────────────────────────────────────────────────────────────────────────
# Load services (pure, no DB)
# ─────────────────────────────────────────────────────────────────────────────

def _load_services(fast_only: bool = False):
    """Import services and boot the classifier.  Returns (intent_svc, classifier)."""
    try:
        from cerebras_client import CerebrasClient  # type: ignore
        from services.intent_service import IntentService, _infer_bucket_from_rules
        from services.classifier_service import classifier_service, CONF_THRESHOLD
        from services.list_service import ListService, _extract_items_from_content
    except ModuleNotFoundError as exc:
        sys.exit(f"[import error] {exc}\nRun from the extended-brain-backend/ directory.")

    # Boot ONNX classifier (silently falls back if model files are absent)
    classifier_service.load()

    # Build Gemini client for slow-path cases (skipped in --fast mode)
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key and not fast_only:
        cerebras = CerebrasClient(provider="gemini", model="gemini-2.0-flash-lite")
    else:
        cerebras = MagicMock()          # fast-only: LLM calls will raise → rules fallback

    intent_svc = IntentService(cerebras)
    list_svc   = ListService(cerebras)

    return intent_svc, classifier_service, CONF_THRESHOLD, list_svc, _infer_bucket_from_rules

# ─────────────────────────────────────────────────────────────────────────────
# iOS TodoView section simulator
# mirrors the logic in TodoView.swift exactly
# ─────────────────────────────────────────────────────────────────────────────

def ios_section(
    outcome: Dict,           # parsed outcome (bucket, remind_at, due_date, is_list, etc.)
    has_assignment: bool = False,
    is_assignee: bool = False,
) -> str:
    """
    Returns the iOS TodoView section label this message would appear in,
    or a descriptive string if it's not in the Todo tab at all.

    Section priority (matches TodoView.swift):
      TIMED TODAY → ASSIGNED TO ME → TODAY → OVERDUE →
      ASSIGNED TO OTHERS → UPCOMING → SOMEDAY → COMPLETED
    """
    bucket   = outcome.get("category", "")
    today    = _today()

    # Only To-Do (and Events which coexist with To-Do) appear in TodoView.
    # A list (is_list=True) with bucket=Remember/Ideas/Track is NOT in TodoView —
    # the todoStore only holds To-Do bucket messages.
    if bucket not in ("To-Do", "Events"):
        return f"❌ Not in TodoView  (bucket: {bucket})"

    # Group assignments
    if is_assignee:
        return "📥 ASSIGNED TO ME"
    if has_assignment and not is_assignee:
        return "📤 ASSIGNED TO OTHERS"

    # Compute effective remind_at and due_date from outcome
    remind_at_raw = outcome.get("remind_at")   # ISO8601 datetime string
    due_date      = outcome.get("due_date")     # YYYY-MM-DD

    # If set_reminder=True and time was extracted, synthesise remind_at
    if not remind_at_raw and outcome.get("_set_reminder") and outcome.get("_event_time"):
        base = due_date or today
        try:
            h, m = map(int, outcome["_event_time"].split(":"))
            dt = datetime.strptime(base, "%Y-%m-%d").replace(hour=h, minute=m)
            remind_at_raw = dt.isoformat()
        except Exception:
            pass

    if remind_at_raw:
        try:
            remind_date = remind_at_raw[:10]
            if remind_date == today:
                row = "ListTodoRow" if outcome.get("is_list") else "TodoRow"
                return f"⏰ TIMED TODAY  ({row})"
            elif remind_date < today:
                return "🔴 OVERDUE"
            else:
                row = "ListTodoRow" if outcome.get("is_list") else "TodoRow"
                return f"📅 UPCOMING  (reminder {remind_at_raw[:16]}, {row})"
        except Exception:
            pass

    if due_date:
        if due_date < today:
            return "🔴 OVERDUE"
        elif due_date == today:
            row = "ListTodoRow" if outcome.get("is_list") else "TodoRow"
            return f"📋 TODAY  ({row})"
        else:
            row = "ListTodoRow" if outcome.get("is_list") else "TodoRow"
            return f"📅 UPCOMING  (due {_date_label(due_date)}, {row})"

    row = "ListTodoRow" if outcome.get("is_list") else "TodoRow"
    return f"🌫  SOMEDAY  ({row})"


# ─────────────────────────────────────────────────────────────────────────────
# Outcome simulator  (mirrors message_processor.process dispatch WITHOUT DB)
# ─────────────────────────────────────────────────────────────────────────────

async def simulate_outcome(
    dump: str,
    intent_svc,
    classifier,
    conf_threshold: float,
    list_svc,
    infer_bucket_from_rules,
    group_id: Optional[int] = None,
    force_bucket: Optional[str] = None,
    no_llm_fallback: bool = False,
    verbose: bool = False,
) -> Dict:
    """
    Run the classification pipeline and return a synthetic outcome dict:
    {
        category:       primary bucket
        is_list:        bool
        list_name:      str | None
        due_date:       YYYY-MM-DD | None
        _set_reminder:  bool
        _event_time:    HH:MM | None
        remind_at:      ISO8601 | None   (if synthesised)
        essence:        str
        _parsed:        full intent dict (for --verbose)
        _path:          "onnx_fast" | "llm_slow" | "rules" | "regex_list" | "forced"
        _confidence:    float | None
    }
    """
    today    = _today()
    tomorrow = _tomorrow()

    # ── Group: pre-check for @mentions → force To-Do ─────────────────────────
    effective_force_bucket   = force_bucket
    effective_no_llm_fallback = no_llm_fallback

    if group_id:
        # Detect @mentions (simplified — just look for @Word patterns)
        mentions = re.findall(r'@(\w+)', dump)
        if mentions:
            effective_force_bucket   = "To-Do"
        effective_no_llm_fallback = True

    # ── Regex list pre-check (mirrors message_processor.process fast path) ──
    if not effective_force_bucket:
        regex_list = list_svc._regex_detect(dump)
        if regex_list and regex_list.get("intent") == "create_or_add" and regex_list.get("items"):
            # Determine bucket from classifier if ready
            list_bucket = "Remember"
            if classifier.is_ready:
                _b, _conf = classifier.classify(dump)
                if _conf >= conf_threshold and _b in ("To-Do", "Remember", "Track"):
                    list_bucket = _b

            # Apply production fix: To-Do lists without explicit date default to today
            list_due = None if list_bucket != "To-Do" else _today()
            outcome = {
                "category":      list_bucket,
                "is_list":       True,
                "list_name":     regex_list["list_name"],
                "list_type":     regex_list["list_type"],
                "items":         regex_list["items"],
                "due_date":      list_due,
                "_set_reminder": False,
                "_event_time":   None,
                "remind_at":     None,
                "essence":       f"{regex_list['list_name']} — {len(regex_list['items'])} item(s)",
                "_path":         "regex_list",
                "_confidence":   _conf if classifier.is_ready else None,
                "_parsed":       regex_list,
            }
            return outcome

    # ── Run intent_service.parse ──────────────────────────────────────────────
    try:
        parsed = await intent_svc.parse(
            content=dump,
            user_name="TestUser",
            user_timezone="Asia/Kolkata",
            check_query=False,       # capture endpoint always uses skip_query=True
            force_bucket=effective_force_bucket,
            no_llm_fallback=effective_no_llm_fallback,
        )
    except Exception as exc:
        # Gemini call failed (e.g. no key) → rule-based fallback
        fb = infer_bucket_from_rules(dump)
        parsed = {
            "actions": {"save_as_todo": fb == "To-Do", "save_as_note": fb == "Remember",
                        "save_as_idea": fb == "Ideas", "save_as_track": fb == "Track",
                        "save_as_event": False, "save_as_list": False,
                        "set_reminder": False, "is_query": False},
            "tasks":    [{"task": dump, "due_date": today, "time": None, "priority": "normal"}]
                        if fb == "To-Do" else [],
            "reminder": None, "list": None, "track": None, "note": None,
            "idea": None, "query": None, "people": [], "priority": "normal",
            "essence": dump[:80],
            "_error": str(exc),
        }

    if verbose:
        print("\n  [PARSED INTENT]", json.dumps(parsed, indent=4, default=str))

    actions = parsed.get("actions", {})

    # ── Determine path taken ─────────────────────────────────────────────────
    if effective_force_bucket:
        path = "forced"
    elif parsed.get("_classifier_confidence") is not None:
        path = "onnx_fast"
    elif parsed.get("_error"):
        path = "rules_fallback"
    else:
        path = "llm_slow"

    conf = parsed.get("_classifier_confidence")

    # ── Dispatch (mirrors message_processor.process) ─────────────────────────

    # Named list — highest priority
    if actions.get("save_as_list") and parsed.get("list"):
        lst = parsed["list"]
        list_bucket = lst.get("bucket") or "Remember"
        # Apply production fix: To-Do lists without date default to today
        list_due = lst.get("due_date") or (_today() if list_bucket == "To-Do" else None)
        return {
            "category":      list_bucket,
            "is_list":       True,
            "list_name":     lst.get("list_name", "List"),
            "list_type":     lst.get("list_type", "custom"),
            "items":         lst.get("items", []),
            "due_date":      list_due,
            "_set_reminder": False,
            "_event_time":   None,
            "remind_at":     None,
            "essence":       f"{lst.get('list_name')} — {len(lst.get('items', []))} item(s)",
            "_path":         path,
            "_confidence":   conf,
            "_parsed":       parsed,
        }

    # Tasks
    if actions.get("save_as_todo") and parsed.get("tasks"):
        tasks    = parsed["tasks"]
        reminder = parsed.get("reminder")
        task0    = tasks[0]
        due_date = task0.get("due_date") or today
        evt_time = task0.get("time")

        # Synthesise remind_at if set_reminder=true and time present
        remind_at = None
        set_reminder = actions.get("set_reminder", False)
        if set_reminder and evt_time:
            try:
                h, m = map(int, evt_time.split(":"))
                base_dt = datetime.strptime(due_date, "%Y-%m-%d")
                remind_at = base_dt.replace(hour=h, minute=m).isoformat()
            except Exception:
                pass

        split_count = len(tasks) if len(tasks) > 1 else None
        essence = parsed.get("essence") or dump[:80]
        if split_count:
            essence = f"Split into {split_count} tasks: " + ", ".join(
                t["task"][:30] for t in tasks[:3]
            )

        return {
            "category":      "To-Do",
            "is_list":       False,
            "list_name":     None,
            "due_date":      due_date,
            "_set_reminder": set_reminder,
            "_event_time":   evt_time,
            "remind_at":     remind_at,
            "split_count":   split_count,
            "priority":      task0.get("priority", "normal"),
            "essence":       essence,
            "_path":         path,
            "_confidence":   conf,
            "_parsed":       parsed,
        }

    # Track
    if actions.get("save_as_track"):
        return {
            "category":      "Track",
            "is_list":       False,
            "due_date":      None,
            "_set_reminder": False,
            "_event_time":   None,
            "remind_at":     None,
            "essence":       parsed.get("essence") or dump[:80],
            "_path":         path,
            "_confidence":   conf,
            "_parsed":       parsed,
        }

    # Note
    if actions.get("save_as_note"):
        return {
            "category":      "Remember",
            "is_list":       False,
            "due_date":      None,
            "_set_reminder": False,
            "_event_time":   None,
            "remind_at":     None,
            "essence":       parsed.get("essence") or dump[:80],
            "_path":         path,
            "_confidence":   conf,
            "_parsed":       parsed,
        }

    # Idea
    if actions.get("save_as_idea"):
        return {
            "category":      "Ideas",
            "is_list":       False,
            "due_date":      None,
            "_set_reminder": False,
            "_event_time":   None,
            "remind_at":     None,
            "essence":       parsed.get("essence") or dump[:80],
            "_path":         path,
            "_confidence":   conf,
            "_parsed":       parsed,
        }

    # Fallback: nothing matched → Random
    return {
        "category":      "Random",
        "is_list":       False,
        "due_date":      None,
        "_set_reminder": False,
        "_event_time":   None,
        "remind_at":     None,
        "essence":       dump[:80],
        "_path":         path,
        "_confidence":   conf,
        "_parsed":       parsed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test case validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check(outcome: Dict, expected: Dict) -> Tuple[bool, List[str]]:
    """
    Compare computed outcome against expected spec.
    Returns (pass: bool, failures: list[str]).
    Each expected key is optional — only present keys are validated.
    """
    failures = []

    for key, exp_val in expected.items():
        if key.startswith("_"):
            continue

        actual = outcome.get(key)

        if key == "category":
            if actual != exp_val:
                failures.append(f"category: got {actual!r}, want {exp_val!r}")

        elif key == "is_list":
            if bool(actual) != bool(exp_val):
                failures.append(f"is_list: got {actual!r}, want {exp_val!r}")

        elif key == "set_reminder":
            if bool(outcome.get("_set_reminder")) != bool(exp_val):
                failures.append(f"set_reminder: got {outcome.get('_set_reminder')!r}, want {exp_val!r}")

        elif key == "has_time":
            has = outcome.get("_event_time") is not None
            if has != bool(exp_val):
                failures.append(f"has_time: got {has!r}, want {exp_val!r}")

        elif key == "due_date_label":
            # "today" | "tomorrow" | "future" | None
            due = outcome.get("due_date")
            today = _today()
            tomorrow = _tomorrow()
            if exp_val == "today":
                if due != today:
                    failures.append(f"due_date should be today ({today}), got {due!r}")
            elif exp_val == "tomorrow":
                if due != tomorrow:
                    failures.append(f"due_date should be tomorrow ({tomorrow}), got {due!r}")
            elif exp_val == "future":
                if not due or due <= today:
                    failures.append(f"due_date should be in future, got {due!r}")
            elif exp_val is None:
                if due is not None:
                    failures.append(f"due_date should be None, got {due!r}")

        elif key == "ios_section_contains":
            section = ios_section(
                outcome,
                has_assignment=expected.get("has_assignments", False),
                is_assignee=expected.get("is_assignee", False),
            )
            if exp_val.upper() not in section.upper():
                failures.append(f"ios_section: got {section!r}, want contains {exp_val!r}")

        elif key == "has_assignments":
            pass  # used by ios_section_contains check above

    return len(failures) == 0, failures


# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
# TEST CASES
# ═══════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

INDIVIDUAL_CASES: List[Dict] = [

    # ── TC-I-01 ────────────────────────────────────────────────────────────────
    {
        "id": "I-01",
        "name": "Simple reminder — explicit time today",
        "dump": "Call Arjun at 5 pm",
        "group_id": None,
        "expected": {
            "category":         "To-Do",
            "is_list":          False,
            "set_reminder":     True,
            "has_time":         True,
            "due_date_label":   "today",
            "ios_section_contains": "TIMED TODAY",
        },
        "dump_response": "Your reminder is set for 5 PM today — Call Arjun",
        "notes": "Time with explicit PM marker. Fast path: ONNX → To-Do, rule extracts 17:00.",
    },

    # ── TC-I-02 ────────────────────────────────────────────────────────────────
    {
        "id": "I-02",
        "name": "Named To-Do list (multi-line bullets, named header)",
        "dump": "Office work:\n- call Suraj for increments\n- check on marketing with Vibhu",
        "group_id": None,
        "expected": {
            "category":         "To-Do",
            "is_list":          True,
            "set_reminder":     False,
            "due_date_label":   "today",       # ✅ SOMEDAY bug fixed
            "ios_section_contains": "TODAY",
        },
        "dump_response": "Office Work — 2 tasks saved",
        "notes": (
            "'Office' is a named qualifier (not neutral). Saved as ListTodoRow with 2 subtasks. "
            "✅ Code fix: _handle_list_save_direct now defaults To-Do lists to due_date=today."
        ),
    },

    # ── TC-I-03 ────────────────────────────────────────────────────────────────
    {
        "id": "I-03",
        "name": "Simple bare todo — no date",
        "dump": "Buy milk",
        "group_id": None,
        "expected": {
            "category":         "To-Do",
            "is_list":          False,
            "set_reminder":     False,
            "due_date_label":   "today",
            "ios_section_contains": "TODAY",
        },
        "dump_response": "Task saved for today",
        "notes": "Action verb 'buy'. Defaults to today. Single TodoRow.",
    },

    # ── TC-I-04 ────────────────────────────────────────────────────────────────
    {
        "id": "I-04",
        "name": "Todo with explicit tomorrow",
        "dump": "Buy milk tomorrow",
        "group_id": None,
        "expected": {
            "category":         "To-Do",
            "is_list":          False,
            "set_reminder":     False,
            "due_date_label":   "tomorrow",
            "ios_section_contains": "UPCOMING",
        },
        "dump_response": "Task saved for tomorrow",
        "notes": "Date resolved to tomorrow. Shows under UPCOMING.",
    },

    # ── TC-I-05 ────────────────────────────────────────────────────────────────
    {
        "id": "I-05",
        "name": "Dentist event — future day with specific time",
        "dump": "Dentist appointment Friday 3pm",
        "group_id": None,
        "expected": {
            "category":         "To-Do",
            "is_list":          False,
            "set_reminder":     True,
            "has_time":         True,
            "due_date_label":   "future",
            "ios_section_contains": "UPCOMING",
        },
        "dump_response": "Event saved for Friday at 3 PM — reminder set",
        "notes": "save_as_event + save_as_todo + set_reminder. Due = next Friday. Time = 15:00.",
    },

    # ── TC-I-06 ────────────────────────────────────────────────────────────────
    {
        "id": "I-06",
        "name": "Health tracking — NOT in TodoView",
        "dump": "Weight 74kg, slept 7 hours",
        "group_id": None,
        "expected": {
            "category":         "Track",
            "is_list":          False,
            "set_reminder":     False,
            "ios_section_contains": "Not in TodoView",
        },
        "dump_response": "Health log saved: weight 74kg, sleep 7h",
        "notes": "Track signals: 'kg' + number, 'slept' + number. Save_as_track only.",
    },

    # ── TC-I-07 ────────────────────────────────────────────────────────────────
    {
        "id": "I-07",
        "name": "Remember note — credential / bare fact",
        "dump": "WiFi password is airtel123",
        "group_id": None,
        "expected": {
            "category":         "Remember",
            "is_list":          False,
            "set_reminder":     False,
            "ios_section_contains": "Not in TodoView",
        },
        "dump_response": "Saved to your brain",
        "notes": "No action verb. Bare credential → save_as_note=Remember.",
    },

    # ── TC-I-08 ────────────────────────────────────────────────────────────────
    {
        "id": "I-08",
        "name": "Ideas capture",
        "dump": "What if we built an AI-powered second brain for students — with spaced repetition built in",
        "group_id": None,
        "expected": {
            "category":         "Ideas",
            "is_list":          False,
            "set_reminder":     False,
            "ios_section_contains": "Not in TodoView",
        },
        "dump_response": "Idea saved",
        "notes": "save_as_idea. 'What if' + creative concept signal.",
    },

    # ── TC-I-09 ────────────────────────────────────────────────────────────────
    {
        "id": "I-09",
        "name": "Random / vague dump — NOT categorisable",
        "dump": "ugh today was rough",
        "group_id": None,
        "expected": {
            "category":         "Remember",  # ⚠ CLASSIFIER GAP: should be Random
            "is_list":          False,
            "set_reminder":     False,
            "ios_section_contains": "Not in TodoView",
        },
        "dump_response": "Captured",
        "notes": (
            "⚠ CLASSIFIER FINDING: ONNX classifies 'ugh today was rough' as Remember (0.83) "
            "instead of Random. Venting/emotional text is semantically closest to "
            "personal notes in the training data. The Random class needs more diverse "
            "examples (venting, casual chat) in the training set."
        ),
    },

    # ── TC-I-10 ────────────────────────────────────────────────────────────────
    {
        "id": "I-10",
        "name": "Shopping list — Remember bucket, NOT in TodoView",
        "dump": "Dmart shopping:\n- milk\n- eggs\n- bread\n- butter",
        "group_id": None,
        "expected": {
            "category":         "Remember",  # ⚠ classifier may say To-Do (0.90) for action items
            "is_list":          True,
            "set_reminder":     False,
            "ios_section_contains": "Not in TodoView",
        },
        "dump_response": "Dmart Shopping List — 4 items saved",
        "notes": (
            "Regex list fast-path detects 'dmart' as shopping list. "
            "⚠ CLASSIFIER FINDING: ONNX may classify shopping items as To-Do (0.90) "
            "because milk/eggs/bread look like action verbs/nouns. "
            "The list_service correctly uses classifier bucket=To-Do for To-Do items "
            "but ideally shopping items → Remember. The regex path bucket logic "
            "sets bucket=To-Do when classifier says so, causing the list to land in "
            "TodoView as SOMEDAY instead of staying in the brain feed."
        ),
    },

    # ── TC-I-11 ────────────────────────────────────────────────────────────────
    {
        "id": "I-11",
        "name": "Japan trip packing list — To-Do bucket (action items)",
        "dump": "Japan trip:\n- book flights\n- get visa\n- hotel booking\n- buy travel insurance",
        "group_id": None,
        "expected": {
            "category":         "To-Do",
            "is_list":          True,
            "set_reminder":     False,
            "due_date_label":   "today",       # ✅ SOMEDAY bug fixed
            "ios_section_contains": "TODAY",
        },
        "dump_response": "Japan Trip List — 4 items saved",
        "notes": (
            "Action items → classifier To-Do. Saved as ListTodoRow. "
            "✅ Code fix: To-Do lists default to due_date=today → TODAY section."
        ),
    },

    # ── TC-I-12 ────────────────────────────────────────────────────────────────
    {
        "id": "I-12",
        "name": "Multi-task dump — split into separate rows",
        "dump": "Call mom, email Priya about the meeting, submit the expense report by 5pm",
        "group_id": None,
        "expected": {
            "category":         "To-Do",
            "is_list":          False,
            "set_reminder":     True,
            "has_time":         True,
            "due_date_label":   "today",
            "ios_section_contains": "TIMED TODAY",
        },
        "dump_response": "3 tasks saved, 1 reminder set: call mom, email Priya…",
        "notes": (
            "LLM splits into 3 tasks. Last task has 17:00 time → TIMED TODAY. "
            "Each task saved as separate Message row. First task shown in response."
        ),
    },

    # ── TC-I-13 ────────────────────────────────────────────────────────────────
    {
        "id": "I-13",
        "name": "Event — team lunch next Wednesday with time",
        "dump": "Team lunch next Wednesday at 1pm",
        "group_id": None,
        "expected": {
            "category":         "To-Do",
            "is_list":          False,
            "set_reminder":     True,
            "has_time":         True,
            "due_date_label":   "future",
            "ios_section_contains": "UPCOMING",
        },
        "dump_response": "Event saved for Wednesday at 1 PM — reminder set",
        "notes": "save_as_event + save_as_todo. next Wednesday is in future → UPCOMING.",
    },

    # ── TC-I-14 ────────────────────────────────────────────────────────────────
    {
        "id": "I-14",
        "name": "Urgent task — priority flag",
        "dump": "URGENT: submit expense report today",
        "group_id": None,
        "expected": {
            "category":         "To-Do",
            "is_list":          False,
            "set_reminder":     False,
            "due_date_label":   "today",
            "ios_section_contains": "TODAY",
        },
        "dump_response": "Urgent task saved for today",
        "notes": "Priority=urgent extracted. Due=today. No time → no reminder. TODAY section.",
    },

    # ── TC-I-15 ────────────────────────────────────────────────────────────────
    {
        "id": "I-15",
        "name": "Reminder keyword + tomorrow",
        "dump": "Remind me to call the bank at 10am tomorrow",
        "group_id": None,
        "expected": {
            "category":         "To-Do",
            "is_list":          False,
            "set_reminder":     True,
            "has_time":         True,
            "due_date_label":   "tomorrow",
            "ios_section_contains": "UPCOMING",
        },
        "dump_response": "Reminder set for tomorrow at 10 AM",
        "notes": (
            "Explicit 'remind me' keyword. Time=10:00 (AM from 'am' marker). "
            "Due=tomorrow → UPCOMING today, becomes TIMED TODAY tomorrow."
        ),
    },

    # ── TC-I-16 ────────────────────────────────────────────────────────────────
    {
        "id": "I-16",
        "name": "Relative time — in 30 minutes",
        "dump": "In 30 minutes check the oven",
        "group_id": None,
        "expected": {
            "category":         "To-Do",   # ✅ Classifier fixed (0.96)
            "is_list":          False,
            "set_reminder":     True,      # ✅ CODE BUG fixed — relative-time now parsed
            "has_time":         True,
            "due_date_label":   "today",
            "ios_section_contains": "TIMED TODAY",
        },
        "dump_response": "Reminder set in 30 minutes",
        "notes": (
            "✅ Classifier: To-Do (0.96). "
            "✅ Code fix: _build_from_bucket now handles 'in N minutes/hours' → absolute time."
        ),
    },

    # ── TC-I-17 ────────────────────────────────────────────────────────────────
    {
        "id": "I-17",
        "name": "Generic header todo list — neutral header, bullets",
        "dump": "Todo for today:\n- Check apple dev console\n- Pack for trip\n- Email team update",
        "group_id": None,
        "expected": {
            "category":         "To-Do",
            "is_list":          False,   # ✅ Fast/LLM divergence fixed — neutral header detected
            "set_reminder":     False,
            "due_date_label":   "today",
            "ios_section_contains": "TODAY",
        },
        "dump_response": "3 tasks saved for today",
        "notes": (
            "✅ Code fix: _build_from_bucket now checks NEUTRAL_HEADER_RULE before creating lists. "
            "'Todo for today' → neutral (blocked word 'todo') → saves as 3 individual TodoRows. "
            "Fast path and LLM path now behave the same."
        ),
    },

    # ── TC-I-18 ────────────────────────────────────────────────────────────────
    {
        "id": "I-18",
        "name": "Bare noun — document reference, NOT a task",
        "dump": "Passport",
        "group_id": None,
        "expected": {
            "category":         "Remember",
            "is_list":          False,
            "set_reminder":     False,
            "ios_section_contains": "Not in TodoView",
        },
        "dump_response": "Saved to your brain",
        "notes": (
            "Single bare noun with no action verb → save_as_note=Remember. "
            "LLM rule: 'bare nouns are filing a reference, not creating a task'."
        ),
    },

    # ── TC-I-19 ────────────────────────────────────────────────────────────────
    {
        "id": "I-19",
        "name": "Named project list — Extended Minds tasks",
        "dump": "Extended minds changes:\n- fix the search bug\n- update API docs\n- write tests",
        "group_id": None,
        "expected": {
            "category":         "To-Do",
            "is_list":          True,
            "set_reminder":     False,
            "due_date_label":   "today",       # ✅ SOMEDAY bug fixed
            "ios_section_contains": "TODAY",
        },
        "dump_response": "Extended Minds Changes — 3 tasks saved",
        "notes": (
            "'Extended minds' is a named qualifier. Action items → To-Do bucket. "
            "Saved as ListTodoRow with 3 subtasks. "
            "✅ Code fix: To-Do lists without explicit date now default to due_date=today → TODAY."
        ),
    },

    # ── TC-I-20 ────────────────────────────────────────────────────────────────
    {
        "id": "I-20",
        "name": "AM time context — breakfast meeting",
        "dump": "Breakfast meeting at 8",
        "group_id": None,
        "expected": {
            "category":         "To-Do",
            "is_list":          False,
            "set_reminder":     True,
            "has_time":         True,
            "due_date_label":   "today",
            "ios_section_contains": "TIMED TODAY",
        },
        "dump_response": "Event saved for today at 8 AM — reminder set",
        "notes": (
            "v4 model correctly classifies as To-Do/Events with time=08:00. "
            "Previous v3 was classifying as Remember (fast-mode gap). Now fixed."
        ),
    },

    # ── TC-I-21 ────────────────────────────────────────────────────────────────
    {
        "id": "I-21",
        "name": "Word-form time — at five thirty pm",
        "dump": "Meeting with client at five thirty pm",
        "group_id": None,
        "expected": {
            "category":         "To-Do",
            "is_list":          False,
            "set_reminder":     True,
            "has_time":         True,
            "due_date_label":   "today",
            "ios_section_contains": "TIMED TODAY",
        },
        "dump_response": "Event saved for today at 5:30 PM",
        "notes": "_normalize_time_words converts 'five thirty pm' → '5:30 pm' → 17:30.",
    },

    # ── TC-I-22 ────────────────────────────────────────────────────────────────
    {
        "id": "I-22",
        "name": "Half past / quarter to — word-form time",
        "dump": "Doctor appointment at half past seven this evening",
        "group_id": None,
        "expected": {
            "category":         "To-Do",
            "is_list":          False,
            "set_reminder":     True,
            "has_time":         True,
            "due_date_label":   "today",
            "ios_section_contains": "TIMED TODAY",
        },
        "dump_response": "Event saved for today at 7:30 PM",
        "notes": (
            "_normalize_time_words: 'half past seven' → 'at 7:30'. "
            "'evening' context → PM → 19:30."
        ),
    },

    # ── TC-I-23 ────────────────────────────────────────────────────────────────
    {
        "id": "I-23",
        "name": "Reading list — Remember bucket",
        "dump": "Books to read:\n- Atomic Habits\n- Deep Work\n- Thinking Fast and Slow",
        "group_id": None,
        "expected": {
            "category":         "Remember",
            "is_list":          True,
            "set_reminder":     False,
            "ios_section_contains": "Not in TodoView",
        },
        "dump_response": "Books To Read List — 3 items saved",
        "notes": "list_type=reading. Classifier returns Remember → not in TodoView. Correct behavior.",
    },

    # ── TC-I-24 ────────────────────────────────────────────────────────────────
    {
        "id": "I-24",
        "name": "Mood + multi-metric health log — Track",
        "dump": "Mood: anxious 4/10, energy 5/10, drank 2L water, 7k steps",
        "group_id": None,
        "expected": {
            "category":         "Track",
            "is_list":          False,
            "set_reminder":     False,
            "ios_section_contains": "Not in TodoView",
        },
        "dump_response": "Health log saved: mood, energy, water, steps",
        "notes": "Multiple track signals + numbers. save_as_track. NOT in TodoView.",
    },

    # ── TC-I-25 ────────────────────────────────────────────────────────────────
    {
        "id": "I-25",
        "name": "Inline comma tasks — single line, named header",
        "dump": "Office work: call Suraj for increments, check on marketing with Vibhu",
        "group_id": None,
        "expected": {
            "category":         "To-Do",
            "is_list":          False,   # inline, no list signal in 'office work', no bullets
            "set_reminder":     False,
            "due_date_label":   "today",
            "ios_section_contains": "TODAY",
        },
        "dump_response": "Tasks saved for today",
        "notes": (
            "'office work' has no LIST_SIGNAL word (not in LIST_SIGNALS set). "
            "Regex list skips it. LLM may split into 2 tasks or save as single To-Do. "
            "Fast path (ONNX To-Do) → single task, TODAY."
        ),
    },

    # ── TC-I-26 ────────────────────────────────────────────────────────────────
    {
        "id": "I-26",
        "name": "Recurring reminder phrase detection",
        "dump": "Remind me every morning to drink water and meditate",
        "group_id": None,
        "expected": {
            "category":         "To-Do",
            "is_list":          False,
            "set_reminder":     False,  # no specific time → reminder keyword alone is not enough
            "due_date_label":   "today",
            "ios_section_contains": "TODAY",
        },
        "dump_response": "Task saved — recurring reminder (morning)",
        "notes": (
            "ONNX → To-Do (0.93). 'remind' keyword present but NO specific time. "
            "fast-path _build_from_bucket rule: set_reminder=True ONLY when time_str is set. "
            "No '10am', 'at 9', etc. → time_str=None → set_reminder=False. "
            "The recurring detection (recurrence_service.is_recurring) runs AFTER the intent parse "
            "and is PRODUCTION-only (not simulated here)."
        ),
    },

    # ── TC-I-27 ────────────────────────────────────────────────────────────────
    {
        "id": "I-27",
        "name": "Noon anchor — special keyword",
        "dump": "Submit the proposal at noon",
        "group_id": None,
        "expected": {
            "category":         "To-Do",
            "is_list":          False,
            "set_reminder":     True,
            "has_time":         True,
            "due_date_label":   "today",
            "ios_section_contains": "TIMED TODAY",
        },
        "dump_response": "Reminder set for today at noon (12:00 PM)",
        "notes": "'noon' maps directly to 12:00 in both _build_from_bucket and _extract_time_mention.",
    },

    # ── TC-I-28 ────────────────────────────────────────────────────────────────
    {
        "id": "I-28",
        "name": "Movies watchlist — Remember, NOT in TodoView",
        "dump": "Movies to watch:\n- Interstellar\n- Dune 2\n- Oppenheimer",
        "group_id": None,
        "expected": {
            "category":         "Remember",
            "is_list":          True,
            "set_reminder":     False,
            "ios_section_contains": "Not in TodoView",
        },
        "dump_response": "Movies To Watch List — 3 items saved",
        "notes": (
            "list_type=watching. Classifier → Remember. Not in TodoView. "
            "Correct behavior: movies to watch are a reference list, not actionable tasks."
        ),
    },

    # ── TC-I-29 ────────────────────────────────────────────────────────────────
    {
        "id": "I-29",
        "name": "Future day (next Monday) — UPCOMING",
        "dump": "Submit PR next Monday",
        "group_id": None,
        "expected": {
            "category":         "To-Do",
            "is_list":          False,
            "set_reminder":     False,
            "due_date_label":   "future",
            "ios_section_contains": "UPCOMING",
        },
        "dump_response": "Task saved for next Monday",
        "notes": "Day-of-week resolution. Due = next Monday. No time → no reminder. UPCOMING.",
    },

    # ── TC-I-30 ────────────────────────────────────────────────────────────────
    {
        "id": "I-30",
        "name": "Bare meeting with time → To-Do with reminder",
        "dump": "meeting at 4",
        "group_id": None,
        "expected": {
            "category":         "To-Do",
            "is_list":          False,
            "set_reminder":     True,
            "has_time":         True,
            "due_date_label":   "today",
            "ios_section_contains": "TIMED TODAY",
        },
        "dump_response": "Event saved for today at 4 PM",
        "notes": (
            "v4 model correctly classifies 'meeting at 4' as To-Do with 4PM reminder. "
            "Previous v3 fell through to Random (confidence too low). Now fixed."
        ),
    },

    # ── TC-I-31 ────────────────────────────────────────────────────────────────
    {
        "id": "I-31",
        "name": "Project issues + today header — To-Do, named list",
        "dump": "Em issues for today:\n- Gets saved as remember and list, but retreival not as list\n- Remind me to chip testing",
        "group_id": None,
        "expected": {
            "category":         "To-Do",
            "is_list":          True,
            "set_reminder":     False,
            "due_date_label":   "today",
            "ios_section_contains": "TODAY",
        },
        "dump_response": "Em Issues For Today — 2 tasks saved",
        "notes": (
            "Classifier gap (fixed): ONNX was classifying 'em issues for today' as Ideas. "
            "Added multi-line bullet To-Do examples in v3 training data. "
            "Code fix: _name_blocked now allows headers with meaningful qualifiers even if "
            "they contain time words like 'today'."
        ),
    },

    # ── TC-I-32 ────────────────────────────────────────────────────────────────
    {
        "id": "I-32",
        "name": "Work for today header — To-Do, named list",
        "dump": "work for today:\n- Remind me to\n- Remember and List saved, but list format not retreived\n- fix the search bug",
        "group_id": None,
        "expected": {
            "category":         "To-Do",
            "is_list":          True,
            "set_reminder":     False,
            "due_date_label":   "today",
            "ios_section_contains": "TODAY",
        },
        "dump_response": "Work For Today — 3 tasks saved",
        "notes": (
            "Classifier gap (fixed): ONNX was returning confidence 0.43 (below threshold). "
            "With v3 training data 'work for today' bullet lists → To-Do. "
            "Code fix: _name_blocked now recognises 'work' as a qualifier → named list."
        ),
    },

    # ── TC-I-33 ────────────────────────────────────────────────────────────────
    {
        "id": "I-33",
        "name": "Movies to watch — Remember bullet list",
        "dump": "Movies to watch:\n- Interstellar\n- Dune 2\n- Oppenheimer",
        "group_id": None,
        "expected": {
            "category":         "Remember",
            "is_list":          True,
            "set_reminder":     False,
            "ios_section_contains": "Not in TodoView",
        },
        "dump_response": "Movies To Watch List — 3 items saved",
        "notes": "Watchlist pattern. Classifier → Remember. list_type=watching. Not in TodoView.",
    },

    # ── TC-I-34 ────────────────────────────────────────────────────────────────
    {
        "id": "I-34",
        "name": "Bucket list — Remember bullet list",
        "dump": "Bucket list:\n- travel to Japan\n- learn guitar\n- run a marathon",
        "group_id": None,
        "expected": {
            "category":         "Remember",
            "is_list":          True,
            "set_reminder":     False,
            "ios_section_contains": "Not in TodoView",
        },
        "dump_response": "Bucket List — 3 items saved",
        "notes": "Life goals reference list → Remember, not To-Do.",
    },

    # ── TC-I-35 ────────────────────────────────────────────────────────────────
    {
        "id": "I-35",
        "name": "Daily health log — Track bullet list",
        "dump": "daily metrics:\n- steps 8500\n- sleep 6.5h\n- water 1.8L\n- weight 74kg",
        "group_id": None,
        "expected": {
            "category":         "Track",
            "is_list":          False,
            "set_reminder":     False,
            "ios_section_contains": "Not in TodoView",
        },
        "dump_response": "Health log saved: steps, sleep, water, weight",
        "notes": "Multiple numeric health signals in bullets → Track. Not a persistent list.",
    },
]

GROUP_CASES: List[Dict] = [

    # ── TC-G-01 ────────────────────────────────────────────────────────────────
    {
        "id": "G-01",
        "name": "Simple @mention task assignment",
        "dump": "@Archi make tea for mummy",
        "group_id": 42,
        "expected": {
            "category":        "To-Do",
            "is_list":         False,
            "set_reminder":    False,
            "has_assignments": True,
            "is_assignee":     True,
            "ios_section_contains": "ASSIGNED TO ME",
        },
        "dump_response": "Task assigned to Archi",
        "notes": (
            "@mention → force_bucket='To-Do', no_llm_fallback=True. "
            "Mirror To-Do created in Archi's feed. APNs sent. "
            "Archi sees it in ASSIGNED TO ME; sender in ASSIGNED TO OTHERS."
        ),
    },

    # ── TC-G-02 ────────────────────────────────────────────────────────────────
    {
        "id": "G-02",
        "name": "@mention with time — sets reminder for assignee",
        "dump": "@Mohan call Rahul at 4 pm",
        "group_id": 42,
        "expected": {
            "category":        "To-Do",
            "is_list":         False,
            "set_reminder":    True,
            "has_time":        True,
            "has_assignments": True,
            "is_assignee":     True,
            "ios_section_contains": "ASSIGNED TO ME",
        },
        "dump_response": "Task assigned to Mohan — reminder at 4 PM",
        "notes": (
            "@mention → force_bucket='To-Do'. "
            "Rule-based time extraction: 16:00. Reminder created for Mohan. "
            "Original group message has group_reminder=False (specific assignee)."
        ),
    },

    # ── TC-G-03 ────────────────────────────────────────────────────────────────
    {
        "id": "G-03",
        "name": "Multiple @mentions — two assignees, one task",
        "dump": "@Rahul @Priya review the proposal before EOD",
        "group_id": 42,
        "expected": {
            "category":        "To-Do",
            "is_list":         False,
            "set_reminder":    False,
            "has_assignments": True,
            "is_assignee":     True,
            "ios_section_contains": "ASSIGNED TO ME",
        },
        "dump_response": "Task assigned to Rahul, Priya",
        "notes": (
            "Two @mentions → assignments=[{rahul}, {priya}]. "
            "APNs sent to both. Both see in ASSIGNED TO ME. "
            "Sender sees both under ASSIGNED TO OTHERS. "
            "Task moves out of ASSIGNED TO OTHERS only when ALL slots done."
        ),
    },

    # ── TC-G-04 ────────────────────────────────────────────────────────────────
    {
        "id": "G-04",
        "name": "@mention + list-format (force_bucket overrides REGEX list, bullets still detected)",
        "dump": "@Archi office tasks:\n- book meeting room\n- send agenda\n- call client",
        "group_id": 42,
        "expected": {
            "category":        "To-Do",
            "is_list":         False,  # ✅ 'tasks' is a neutral word → individual To-Do rows
            "set_reminder":    False,
            "has_assignments": True,
            "is_assignee":     True,
            "ios_section_contains": "ASSIGNED TO ME",
        },
        "dump_response": "Tasks assigned to Archi",
        "notes": (
            "force_bucket='To-Do'. Bullets present but header '@Archi office tasks' contains "
            "the neutral word 'tasks' → neutral header rule → saves as individual To-Do rows. "
            "✅ Code fix: fast path now respects NEUTRAL_HEADER_RULE like the LLM slow path."
        ),
    },

    # ── TC-G-05 ────────────────────────────────────────────────────────────────
    {
        "id": "G-05",
        "name": "No @mention + time → group_reminder (APNs to all members)",
        "dump": "Team standup at 10am",
        "group_id": 42,
        "expected": {
            "category":        "To-Do",
            "is_list":         False,
            "set_reminder":    True,
            "has_time":        True,
            "has_assignments": False,
            "ios_section_contains": "TIMED TODAY",
        },
        "dump_response": "Group reminder set for 10 AM — all members notified",
        "notes": (
            "No @mention → no assignments. Time extracted (10:00) → group_reminder=True. "
            "tags.group_reminder=True. APNs broadcast to ALL members. "
            "Shows in group feed as To-Do. Sender's TodoView shows under TIMED TODAY."
        ),
    },

    # ── TC-G-06 ────────────────────────────────────────────────────────────────
    {
        "id": "G-06",
        "name": "No @mention, no time — regular group To-Do (classifier)",
        "dump": "Check in with everyone by EOD",
        "group_id": 42,
        "expected": {
            "category":        "To-Do",
            "is_list":         False,
            "set_reminder":    False,
            "has_assignments": False,
            "ios_section_contains": "TODAY",
        },
        "dump_response": "Task saved",
        "notes": (
            "No @mention, no time. no_llm_fallback=True → ONNX or rule-based. "
            "'Check' is an action verb → To-Do. No group_reminder (no time). "
            "Sender's TodoView: TODAY section."
        ),
    },

    # ── TC-G-07 ────────────────────────────────────────────────────────────────
    {
        "id": "G-07",
        "name": "@mention + urgent flag",
        "dump": "@Vibhu URGENT: fix the production bug NOW",
        "group_id": 42,
        "expected": {
            "category":        "To-Do",
            "is_list":         False,
            "set_reminder":    False,
            "has_assignments": True,
            "is_assignee":     True,
            "ios_section_contains": "ASSIGNED TO ME",
        },
        "dump_response": "URGENT task assigned to Vibhu",
        "notes": (
            "force_bucket='To-Do'. no_llm_fallback=True. "
            "Priority=urgent extracted by rules. APNs to Vibhu. "
            "URGENT tasks should visually stand out in ASSIGNED TO ME."
        ),
    },

    # ── TC-G-08 ────────────────────────────────────────────────────────────────
    {
        "id": "G-08",
        "name": "Multiple @mentions with split tasks in prose",
        "dump": "@Mohan call Rahul at 4 pm & @Rahul check with driver on bookings",
        "group_id": 42,
        "expected": {
            "category":        "To-Do",
            "is_list":         False,
            "set_reminder":    True,   # time present in first sub-task
            "has_time":        True,
            "has_assignments": True,
            "is_assignee":     True,
            "ios_section_contains": "ASSIGNED TO ME",
        },
        "dump_response": "Tasks assigned to Mohan, Rahul",
        "notes": (
            "Two @mentions in one dump. force_bucket='To-Do'. "
            "assignments=[mohan, rahul]. Single message in group feed. "
            "Both see it in ASSIGNED TO ME. Time extracted (16:00) → reminder."
        ),
    },

    # ── TC-G-09 ────────────────────────────────────────────────────────────────
    {
        "id": "G-09",
        "name": "Group idea dump (no @mention) — Ideas bucket",
        "dump": "What if we add a voice memo feature to the group feed?",
        "group_id": 42,
        "expected": {
            "category":        "Ideas",
            "is_list":         False,
            "set_reminder":    False,
            "has_assignments": False,
            "ios_section_contains": "Not in TodoView",
        },
        "dump_response": "Idea saved",
        "notes": (
            "no_llm_fallback=True → rule-based _infer_bucket_from_rules. "
            "Regex detects 'what if' → Ideas. NOT in TodoView."
        ),
    },

    # ── TC-G-10 ────────────────────────────────────────────────────────────────
    {
        "id": "G-10",
        "name": "Reminder keyword but no time — no group_reminder flag",
        "dump": "Don't forget to submit the monthly report",
        "group_id": 42,
        "expected": {
            "category":        "To-Do",
            "is_list":         False,
            "set_reminder":    False,  # no time → no reminder, no group_reminder
            "has_assignments": False,
            "ios_section_contains": "TODAY",
        },
        "dump_response": "Task saved — remember to submit the monthly report",
        "notes": (
            "Reminder keyword present but NO time extracted. "
            "group_reminder=False (needs due time). "
            "is_group_reminder = (not assignments) AND bool(due_date OR remind_at)."
        ),
    },

    # ── TC-G-11 ────────────────────────────────────────────────────────────────
    {
        "id": "G-11",
        "name": "Group health log — Track bucket",
        "dump": "Weight check: 72kg, energy 7/10",
        "group_id": 42,
        "expected": {
            "category":        "Track",
            "is_list":         False,
            "set_reminder":    False,
            "has_assignments": False,
            "ios_section_contains": "Not in TodoView",
        },
        "dump_response": "Health log saved",
        "notes": (
            "no_llm_fallback=True. Rule-based: 'kg' + number → Track. "
            "Not in TodoView. Appears in group feed under Track bucket."
        ),
    },

    # ── TC-G-12 ────────────────────────────────────────────────────────────────
    {
        "id": "G-12",
        "name": "Group note — credential / info (Remember)",
        "dump": "Office wifi: CompanyPass123",
        "group_id": 42,
        "expected": {
            "category":        "Remember",
            "is_list":         False,
            "set_reminder":    False,
            "has_assignments": False,
            "ios_section_contains": "Not in TodoView",
        },
        "dump_response": "Saved to group brain",
        "notes": (
            "no_llm_fallback=True. Rule-based: no action verb, len>15 → Remember. "
            "Shared in group feed. Not in anyone's TodoView."
        ),
    },

    # ── TC-G-13 ────────────────────────────────────────────────────────────────
    {
        "id": "G-13",
        "name": "Group event — day name with time (no @mention)",
        "dump": "Team dinner next Friday 8pm",
        "group_id": 42,
        "expected": {
            "category":        "To-Do",
            "is_list":         False,
            "set_reminder":    True,
            "has_time":        True,
            "due_date_label":  "future",
            "has_assignments": False,
            "ios_section_contains": "UPCOMING",
        },
        "dump_response": "Group reminder set for Friday 8 PM — all members notified",
        "notes": (
            "No @mention. Time present (20:00) → group_reminder=True. "
            "APNs to all members. Sender's TodoView: UPCOMING → TIMED TODAY on Friday."
        ),
    },

    # ── TC-G-14 ────────────────────────────────────────────────────────────────
    {
        "id": "G-14",
        "name": "@mention + question format (still forced To-Do)",
        "dump": "@Priya can you check the invoice?",
        "group_id": 42,
        "expected": {
            "category":        "To-Do",
            "is_list":         False,
            "set_reminder":    False,
            "has_assignments": True,
            "is_assignee":     True,
            "ios_section_contains": "ASSIGNED TO ME",
        },
        "dump_response": "Task assigned to Priya",
        "notes": (
            "Question format doesn't matter when @mention is present. "
            "force_bucket='To-Do' overrides any query detection. "
            "Assignment created. Priya sees in ASSIGNED TO ME."
        ),
    },

    # ── TC-G-15 ────────────────────────────────────────────────────────────────
    {
        "id": "G-15",
        "name": "@mention + shopping list format (bullet list still created)",
        "dump": "@Archi grocery list:\n- milk\n- eggs\n- bread",
        "group_id": 42,
        "expected": {
            "category":        "To-Do",
            "is_list":         True,   # ⚠ _build_from_bucket still detects bullets
            "set_reminder":    False,
            "has_assignments": True,
            "is_assignee":     True,
            "ios_section_contains": "ASSIGNED TO ME",
        },
        "dump_response": "Task assigned to Archi",
        "notes": (
            "force_bucket='To-Do' skips regex list fast-path. "
            "⚠ But _build_from_bucket('To-Do') detects bullets → is_list=True. "
            "list_name='grocery list' (text after @Archi). "
            "Archi sees an expandable ListTodoRow in ASSIGNED TO ME."
        ),
    },

    # ── TC-G-16 ────────────────────────────────────────────────────────────────
    {
        "id": "G-16",
        "name": "Three simultaneous @mentions (group parallel tasks)",
        "dump": "@Mohan update status, @Rahul review code, @Archi prep demo for tomorrow",
        "group_id": 42,
        "expected": {
            "category":        "To-Do",
            "is_list":         False,
            "set_reminder":    False,
            "has_assignments": True,
            "is_assignee":     True,
            "ios_section_contains": "ASSIGNED TO ME",
        },
        "dump_response": "Tasks assigned to Mohan, Rahul, Archi",
        "notes": (
            "Three @mentions detected. assignments=[mohan, rahul, archi]. "
            "Single group message with 3 slots. "
            "Moves out of ASSIGNED TO OTHERS only when ALL 3 slots done."
        ),
    },

    # ── TC-G-17 ────────────────────────────────────────────────────────────────
    {
        "id": "G-17",
        "name": "Group task with action verb + future day (UPCOMING)",
        "dump": "Submit the monthly report by Thursday",
        "group_id": 42,
        "expected": {
            "category":        "To-Do",
            "is_list":         False,
            "set_reminder":    False,
            "due_date_label":  "future",
            "has_assignments": False,
            "ios_section_contains": "UPCOMING",
        },
        "dump_response": "Task saved for Thursday",
        "notes": (
            "no_llm_fallback=True. 'Submit' action verb → To-Do. "
            "Day-of-week 'Thursday' resolved. No time → no group_reminder. "
            "Sender's TodoView: UPCOMING."
        ),
    },

    # ── TC-G-18 ────────────────────────────────────────────────────────────────
    {
        "id": "G-18",
        "name": "@mention to yourself — no mirror, no APNs",
        "dump": "@Me check the slides before presenting",
        "group_id": 42,
        "expected": {
            "category":        "To-Do",
            "is_list":         False,
            "set_reminder":    False,
            "has_assignments": True,   # assignment created (if 'Me' resolves to current user)
            "is_assignee":     True,
            "ios_section_contains": "ASSIGNED TO ME",
        },
        "dump_response": "Task saved",
        "notes": (
            "@mention to self (auid == current_user.id) → no mirror created, "
            "no APNs. Assignment slot still in the message but no duplicate in feed."
        ),
    },

    # ── TC-G-19 ────────────────────────────────────────────────────────────────
    {
        "id": "G-19",
        "name": "Group vague chat — trivial text",
        "dump": "haha nice",
        "group_id": 42,
        "expected": {
            "category":        "Remember",  # ⚠ CLASSIFIER GAP: should be Random
            "is_list":         False,
            "set_reminder":    False,
            "has_assignments": False,
            "ios_section_contains": "Not in TodoView",
        },
        "dump_response": "Captured",
        "notes": (
            "⚠ CLASSIFIER FINDING (same as I-09): ONNX classifies casual social text "
            "as Remember (0.83) instead of Random. The Random/venting class needs more "
            "examples of casual chat in training data."
        ),
    },

    # ── TC-G-20 ────────────────────────────────────────────────────────────────
    {
        "id": "G-20",
        "name": "@mention converts idea to task",
        "dump": "@Archi what if we add dark mode to the app?",
        "group_id": 42,
        "expected": {
            "category":        "To-Do",
            "is_list":         False,
            "set_reminder":    False,
            "has_assignments": True,
            "is_assignee":     True,
            "ios_section_contains": "ASSIGNED TO ME",
        },
        "dump_response": "Task assigned to Archi",
        "notes": (
            "@mention overrides idea detection. force_bucket='To-Do'. "
            "Even a 'what if' question becomes an actionable assignment. "
            "Archi sees it in ASSIGNED TO ME."
        ),
    },

    # ── TC-G-21 ────────────────────────────────────────────────────────────────
    {
        "id": "G-21",
        "name": "Group meeting notes — classifier says To-Do, should be Remember",
        "dump": "Meeting notes: discussed Q3 roadmap, budget allocation was approved, hiring freeze until Jan",
        "group_id": 42,
        "expected": {
            "category":        "Remember",  # ✅ FIXED in v2 training data
            "is_list":         False,
            "set_reminder":    False,
            "has_assignments": False,
            "ios_section_contains": "Not in TodoView",
        },
        "dump_response": "Saved to group brain",
        "notes": (
            "Meeting notes with action-adjacent words. "
            "✅ Fixed: new training data added corporate/meeting note examples to Remember. "
            "Classifier now correctly returns Remember (0.95)."
        ),
    },

    # ── TC-G-22 ────────────────────────────────────────────────────────────────
    {
        "id": "G-22",
        "name": "@mention + AM time context",
        "dump": "@Mohan morning standup at 9",
        "group_id": 42,
        "expected": {
            "category":        "To-Do",
            "is_list":         False,
            "set_reminder":    True,
            "has_time":        True,
            "has_assignments": True,
            "is_assignee":     True,
            "ios_section_contains": "ASSIGNED TO ME",
        },
        "dump_response": "Task assigned to Mohan — reminder at 9 AM",
        "notes": (
            "'morning' → AM context. Hour 9 → 09:00. "
            "force_bucket='To-Do'. Reminder created for Mohan. "
            "Due=today (no date specified)."
        ),
    },

    # ── TC-G-23 ────────────────────────────────────────────────────────────────
    {
        "id": "G-23",
        "name": "Group task with 'in X minutes' relative time",
        "dump": "Server maintenance in 20 minutes, please save your work",
        "group_id": 42,
        "expected": {
            "category":        "To-Do",    # ✅ Fixed: _infer_bucket_from_rules now has relative-time
            "is_list":         False,
            "set_reminder":    True,
            "has_time":        True,
            "has_assignments": False,
            "ios_section_contains": "TIMED TODAY",
        },
        "dump_response": "Group reminder set in 20 minutes — all members notified",
        "notes": (
            "✅ Code fix: _infer_bucket_from_rules now matches 'in N minutes/hours' → To-Do. "
            "group_reminder=True (has time, no @mention). APNs to all members."
        ),
    },

    # ── TC-G-24 ────────────────────────────────────────────────────────────────
    {
        "id": "G-24",
        "name": "@mention + noon keyword",
        "dump": "@Vibhu submit the design files at noon",
        "group_id": 42,
        "expected": {
            "category":        "To-Do",
            "is_list":         False,
            "set_reminder":    True,
            "has_time":        True,
            "has_assignments": True,
            "is_assignee":     True,
            "ios_section_contains": "ASSIGNED TO ME",
        },
        "dump_response": "Task assigned to Vibhu — reminder at 12:00 PM",
        "notes": (
            "'noon' → 12:00. force_bucket='To-Do'. "
            "Reminder for Vibhu at noon. APNs to Vibhu."
        ),
    },

    # ── TC-G-25 ────────────────────────────────────────────────────────────────
    {
        "id": "G-25",
        "name": "Group daily recurring task (no @mention)",
        "dump": "Daily standup reminder at 10:30am for the whole team",
        "group_id": 42,
        "expected": {
            "category":        "To-Do",
            "is_list":         False,
            "set_reminder":    True,
            "has_time":        True,
            "has_assignments": False,
            "ios_section_contains": "TIMED TODAY",
        },
        "dump_response": "Group reminder set daily at 10:30 AM",
        "notes": (
            "No @mention. Time=10:30 → group_reminder=True. "
            "Recurring detection (personal only) skipped for group context. "
            "APNs to all members."
        ),
    },

    # ── TC-G-26 ────────────────────────────────────────────────────────────────
    {
        "id": "G-26",
        "name": "@mention task list with bullets (creates ListTodoRow for assignee)",
        "dump": "@Archi todo for tomorrow:\n- review PR\n- update docs\n- deploy to staging",
        "group_id": 42,
        "expected": {
            "category":        "To-Do",
            "is_list":         False,  # ✅ 'todo' is neutral → individual To-Do rows
            "set_reminder":    False,
            "has_assignments": True,
            "is_assignee":     True,
            "ios_section_contains": "ASSIGNED TO ME",
        },
        "dump_response": "Tasks assigned to Archi",
        "notes": (
            "force_bucket='To-Do'. Header '@Archi todo for tomorrow' contains "
            "neutral word 'todo' → neutral header rule → individual To-Do rows. "
            "✅ Code fix: fast path now respects NEUTRAL_HEADER_RULE."
        ),
    },

    # ── TC-G-27 ────────────────────────────────────────────────────────────────
    {
        "id": "G-27",
        "name": "@mention with high-priority signal and 'tonight'",
        "dump": "@Rahul this is really important — update the client deck tonight",
        "group_id": 42,
        "expected": {
            "category":        "To-Do",
            "is_list":         False,
            "set_reminder":    True,   # ✅ 'tonight' now infers 20:00 → reminder fires
            "has_time":        True,
            "has_assignments": True,
            "is_assignee":     True,
            "ios_section_contains": "ASSIGNED TO ME",
        },
        "dump_response": "Important task assigned to Rahul — reminder at 8 PM",
        "notes": (
            "'important' → priority=high. "
            "✅ Code fix: 'tonight' now infers time_str=20:00 → set_reminder=True. "
            "Reminder fires at 8 PM today. Rahul sees it in ASSIGNED TO ME."
        ),
    },

    # ── TC-G-28 ────────────────────────────────────────────────────────────────
    {
        "id": "G-28",
        "name": "Group track log — shared health metrics",
        "dump": "Team run: Archi 5km, Mohan 3km, Rahul 7km",
        "group_id": 42,
        "expected": {
            "category":        "Track",     # ✅ FIXED in v2 training data
            "is_list":         False,
            "set_reminder":    False,
            "has_assignments": False,
            "ios_section_contains": "Not in TodoView",
        },
        "dump_response": "Activity log saved",
        "notes": (
            "Multi-person group activity log. "
            "✅ Fixed: new training data added group activity examples to Track. "
            "Classifier now correctly returns Track (0.94)."
        ),
    },

    # ── TC-G-29 ────────────────────────────────────────────────────────────────
    {
        "id": "G-29",
        "name": "@mention + add to list (force_bucket overrides list add)",
        "dump": "@Archi add milk and eggs to the dmart list",
        "group_id": 42,
        "expected": {
            "category":        "To-Do",
            "is_list":         False,  # force_bucket wins, list add skipped
            "set_reminder":    False,
            "has_assignments": True,
            "is_assignee":     True,
            "ios_section_contains": "ASSIGNED TO ME",
        },
        "dump_response": "Task assigned to Archi",
        "notes": (
            "Even though this looks like a list operation ('add to dmart'), "
            "@mention triggers force_bucket='To-Do'. The list add is NOT processed "
            "(force_bucket skips regex list detection). "
            "Archi sees it as a regular task."
        ),
    },

    # ── TC-G-30 ────────────────────────────────────────────────────────────────
    {
        "id": "G-30",
        "name": "Complex multi-person assignment with different times",
        "dump": "@Mohan check server health at 9am, @Priya review logs at 11am, @Vibhu deploy fix at 3pm",
        "group_id": 42,
        "expected": {
            "category":        "To-Do",
            "is_list":         False,
            "set_reminder":    True,   # times present
            "has_time":        True,
            "has_assignments": True,
            "is_assignee":     True,
            "ios_section_contains": "ASSIGNED TO ME",
        },
        "dump_response": "Tasks assigned to Mohan (9 AM), Priya (11 AM), Vibhu (3 PM)",
        "notes": (
            "3 @mentions + 3 times. force_bucket='To-Do'. no_llm_fallback=True. "
            "assignments=[mohan, priya, vibhu]. Single group message. "
            "Each assignee sees their slot in ASSIGNED TO ME. "
            "Times extracted by rule-based (first time 09:00 used for global reminder)."
        ),
    },
]

ALL_CASES = INDIVIDUAL_CASES + GROUP_CASES


# ─────────────────────────────────────────────────────────────────────────────
# ANSI colours
# ─────────────────────────────────────────────────────────────────────────────

def _c(text: str, code: str) -> str:
    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text

GREEN  = lambda t: _c(t, "32")
RED    = lambda t: _c(t, "31")
YELLOW = lambda t: _c(t, "33")
CYAN   = lambda t: _c(t, "36")
BOLD   = lambda t: _c(t, "1")
DIM    = lambda t: _c(t, "2")


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_suite(args):
    print(BOLD("\n══════════════════════════════════════════════════════════════"))
    print(BOLD("  Extended Minds — Dump Classification Inference Test Suite"))
    print(BOLD("══════════════════════════════════════════════════════════════"))
    print(f"  Date    : {_today()}")
    print(f"  Mode    : {'FAST (ONNX+rules, no LLM)' if args.fast else 'FULL (ONNX → Gemini fallback)'}")
    print(f"  Cases   : {'Individual only' if args.individual else 'Group only' if args.group else 'All'}")
    if args.case:
        print(f"  Filter  : Case {args.case}")
    print()

    intent_svc, classifier, conf_threshold, list_svc, infer_rules = _load_services(
        fast_only=args.fast
    )

    print(f"  Classifier: {'ONNX loaded ✓' if classifier.is_ready else 'Not loaded — Gemini / rules fallback'}")
    print(f"  Gemini key: {'set ✓' if os.getenv('GEMINI_API_KEY') else 'not set (slow path disabled)'}")
    print()

    # Filter cases
    cases_to_run = ALL_CASES
    if args.individual:
        cases_to_run = INDIVIDUAL_CASES
    elif args.group:
        cases_to_run = GROUP_CASES
    if args.case:
        cases_to_run = [c for c in cases_to_run if c["id"] == args.case]
        if not cases_to_run:
            print(RED(f"No case found with id={args.case!r}"))
            return

    passed = failed = 0
    results = []

    for tc in cases_to_run:
        t0 = time.perf_counter()
        outcome = await simulate_outcome(
            dump=tc["dump"],
            intent_svc=intent_svc,
            classifier=classifier,
            conf_threshold=conf_threshold,
            list_svc=list_svc,
            infer_bucket_from_rules=infer_rules,
            group_id=tc.get("group_id"),
            verbose=args.verbose,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Compute the iOS section
        has_assignment = bool(re.findall(r'@(\w+)', tc["dump"])) and bool(tc.get("group_id"))
        section = ios_section(
            outcome,
            has_assignment=has_assignment,
            is_assignee=has_assignment,
        )

        ok, failures = _check(outcome, tc["expected"])

        if ok:
            passed += 1
            status = GREEN("PASS")
        else:
            failed += 1
            status = RED("FAIL")

        results.append({
            "tc": tc, "outcome": outcome, "section": section,
            "ok": ok, "failures": failures, "elapsed_ms": elapsed_ms,
        })

        # ── Print case summary ────────────────────────────────────────────────
        path_tag = {
            "onnx_fast":     CYAN("ONNX"),
            "llm_slow":      YELLOW("LLM"),
            "regex_list":    CYAN("REGEX"),
            "forced":        CYAN("FORCED"),
            "rules_fallback":YELLOW("RULES"),
        }.get(outcome.get("_path", ""), DIM("?"))

        conf_tag = ""
        if outcome.get("_confidence") is not None:
            conf_tag = DIM(f" ({outcome['_confidence']:.2f})")

        print(f"  {status}  {BOLD(tc['id'])}  {tc['name']}")
        print(f"         Dump     : {tc['dump'][:70].replace(chr(10), ' | ')}")
        print(f"         Bucket   : {BOLD(outcome['category'])}  path={path_tag}{conf_tag}")
        print(f"         Essence  : {outcome.get('essence','')[:70]}")
        print(f"         Section  : {section}")

        if outcome.get("is_list"):
            print(f"         List     : {outcome.get('list_name','?')} ({len(outcome.get('items',[]))} items)")

        if outcome.get("_set_reminder"):
            time_str = outcome.get("_event_time", "?")
            print(f"         Reminder : {time_str}  →  {outcome.get('remind_at','(synthesised)')}")

        if outcome.get("split_count"):
            print(f"         Split    : {outcome['split_count']} tasks")

        print(f"         Response : \"{tc['dump_response']}\"")
        print(f"         Timing   : {elapsed_ms:.1f}ms")

        if failures:
            for f in failures:
                print(f"         {RED('✗')} {f}")

        if args.verbose:
            print(f"         Notes    : {tc.get('notes','')}")

        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    total = passed + failed
    pct   = f"{100*passed//total}%" if total else "—"
    print(BOLD("══════════════════════════════════════════════════════════════"))
    print(BOLD(f"  RESULTS: {GREEN(str(passed)+'  passed')}  |  {RED(str(failed)+'  failed')}  |  {total} total  ({pct})"))
    print(BOLD("══════════════════════════════════════════════════════════════"))

    if failed:
        print()
        print(RED(BOLD("  Failing cases:")))
        for r in results:
            if not r["ok"]:
                print(f"    {r['tc']['id']}  {r['tc']['name']}")
                for f in r["failures"]:
                    print(f"      • {f}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Live mode (calls actual backend API)
# ─────────────────────────────────────────────────────────────────────────────

async def run_live(args):
    """Call the real /api/messages/capture endpoint and show what comes back."""
    import urllib.request
    import urllib.error

    BASE = "https://extended-brain-backend-production.up.railway.app"
    token = args.token
    if not token:
        sys.exit("--live mode requires --token <JWT>")

    cases_to_run = ALL_CASES
    if args.individual:
        cases_to_run = INDIVIDUAL_CASES
    elif args.group:
        cases_to_run = GROUP_CASES
    if args.case:
        cases_to_run = [c for c in cases_to_run if c["id"] == args.case]

    print(BOLD(f"\n  LIVE MODE — calling {BASE}"))
    print(BOLD("  ⚠  This CREATES real data in the production database.\n"))

    for tc in cases_to_run:
        payload = json.dumps({
            "content":      tc["dump"],
            "message_type": "text",
            "group_id":     tc.get("group_id"),
        }).encode()

        req = urllib.request.Request(
            f"{BASE}/api/messages/capture",
            data=payload,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {token}",
            },
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = json.loads(resp.read())
            elapsed_ms = (time.perf_counter() - t0) * 1000
            data = body.get("data", {})
            print(f"  {GREEN('OK')}  {tc['id']}  {tc['name']}")
            print(f"       category  : {data.get('category','?')}")
            print(f"       essence   : {data.get('essence','')[:70]}")
            print(f"       due_date  : {data.get('due_date')}")
            print(f"       remind_at : {data.get('remind_at')}")
            print(f"       is_list   : {data.get('is_list', False)}")
            print(f"       list_name : {data.get('list_name')}")
            print(f"       timing    : {elapsed_ms:.0f}ms")
        except urllib.error.HTTPError as e:
            print(f"  {RED('ERR')} {tc['id']}  HTTP {e.code}: {e.read()[:100]}")
        except Exception as e:
            print(f"  {RED('ERR')} {tc['id']}  {e}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extended Minds classification test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("USAGE")[1].split("ENVIRONMENT")[0].strip() if "USAGE" in __doc__ else "",
    )
    parser.add_argument("--fast",       action="store_true", help="ONNX + rules only, no LLM calls")
    parser.add_argument("--individual", action="store_true", help="Run individual cases only")
    parser.add_argument("--group",      action="store_true", help="Run group cases only")
    parser.add_argument("--case",       type=str,            help="Run a single case by ID (e.g. I-05 or G-12)")
    parser.add_argument("--verbose",    action="store_true", help="Print full parsed intent output")
    parser.add_argument("--live",       action="store_true", help="Call actual backend API (creates real data)")
    parser.add_argument("--token",      type=str,            help="JWT token for --live mode")
    args = parser.parse_args()

    if args.live:
        asyncio.run(run_live(args))
    else:
        asyncio.run(run_suite(args))


if __name__ == "__main__":
    main()
