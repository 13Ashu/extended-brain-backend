"""
Intent Service — Multi-Action Parser
─────────────────────────────────────────────────────────────
A single message can trigger MULTIPLE actions simultaneously.

Examples:
  "remind me to call mom at 10pm"
    → save_as_todo: true   (it's a task)
    → set_reminder: true   (it needs a reminder)
    → NOT two separate intents — one message, two actions

  "dentist appointment Friday 3pm"
    → save_as_todo: true   (something to act on)
    → save_as_event: true  (it's a scheduled event)
    → set_reminder: true   (remind before it happens)

  "dmart shopping:\n- milk\n- eggs"
    → save_as_list: true   (named list with items)
    → save_as_todo: false  (not a task)
    → set_reminder: false

  "weight 74kg, slept 7 hours"
    → save_as_track: true
    → save_as_todo: false

Output schema — always returns all action flags:
{
  "actions": {
    "save_as_todo":  bool,   # save in To-Do bucket
    "save_as_event": bool,   # save in Events bucket  
    "save_as_note":  bool,   # save in Remember bucket
    "save_as_idea":  bool,   # save in Ideas bucket
    "save_as_track": bool,   # save in Track bucket
    "save_as_list":  bool,   # save as named list
    "set_reminder":  bool,   # create a reminder
    "is_query":      bool,   # user wants to retrieve something
  },
  "tasks": [                 # present when save_as_todo=true
    {
      "task": "clean description",
      "due_date": "YYYY-MM-DD",
      "time": "HH:MM or null",
      "priority": "normal|high|urgent"
    }
  ],
  "reminder": {              # present when set_reminder=true
    "due_date": "YYYY-MM-DD",
    "time": "HH:MM",
    "priority": "normal|high|urgent"
  },
  "event": {                 # present when save_as_event=true
    "title": "short title",
    "due_date": "YYYY-MM-DD",
    "time": "HH:MM or null",
    "people": []
  },
  "list": {                  # present when save_as_list=true
    "list_name": "Dmart Shopping List",
    "list_type": "shopping|bag|packing|reading|watching|custom",
    "items": ["item1", "item2"],
    "due_date": "YYYY-MM-DD or null"
  },
  "track": {                 # present when save_as_track=true
    "logs": [{"metric": "weight", "value": "74", "unit": "kg"}]
  },
  "note": {                  # present when save_as_note=true
    "content": "the fact",
    "keywords": []
  },
  "idea": {                  # present when save_as_idea=true
    "content": "the idea",
    "keywords": []
  },
  "query": {                 # present when is_query=true
    "query_text": "what they want to find",
    "date_hint": "today|tomorrow|this_week|null",
    "list_name": "if asking for a named list, else null"
  },
  "people": [],              # all people mentioned
  "priority": "normal",      # overall message priority
  "essence": "one sentence summary"
}
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from cerebras_client import CerebrasClient

IST = ZoneInfo("Asia/Kolkata")

COMMAND_PREFIXES = {
    "/start", "/link", "/status", "briefing:", "project:",
    "subtask:", "search:", "find:", "get:",
}

def _is_command(content: str) -> bool:
    lc = content.lower().strip()
    return any(lc.startswith(p) for p in COMMAND_PREFIXES)

def _is_trivial(content: str) -> bool:
    lc = content.lower().strip()
    trivials = {"hi", "hey", "heyy", "hello", "ok", "okay", "lol", "test", "hmm", "hm"}
    return len(lc) < 4 or lc in trivials


# ─────────────────────────────────────────────────────────────────────────────
# Default empty result
# ─────────────────────────────────────────────────────────────────────────────

def _empty_actions() -> Dict:
    return {
        "save_as_todo":  False,
        "save_as_event": False,
        "save_as_note":  False,
        "save_as_idea":  False,
        "save_as_track": False,
        "save_as_list":  False,
        "set_reminder":  False,
        "is_query":      False,
    }

def _default_result() -> Dict:
    return {
        "actions":  _empty_actions(),
        "tasks":    [],
        "reminder": None,
        "event":    None,
        "list":     None,
        "track":    None,
        "note":     None,
        "idea":     None,
        "query":    None,
        "people":   [],
        "priority": "normal",
        "essence":  "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Intent Service
# ─────────────────────────────────────────────────────────────────────────────

class IntentService:

    def __init__(self, cerebras_client: CerebrasClient):
        self.cerebras = cerebras_client
        # Gemini Flash Lite — used for enrichment (essence, entities) and low-confidence fallback
        self.fast = CerebrasClient(provider="gemini", model="gemini-2.5-flash-lite")

    async def parse(
        self,
        content: str,
        user_name: str,
        user_timezone: str = "Asia/Kolkata",
        check_query: bool = True,
        force_bucket: Optional[str] = None,
        no_llm_fallback: bool = False,
    ) -> Dict:
        """
        Parse a message into multi-action structured output.

        force_bucket: skip classifier and LLM entirely; build result from this bucket.
        no_llm_fallback: when True, never call Gemini — use rule-based bucket if classifier fails.

        Fast path  (classifier ready, confidence ≥ threshold):
            ONNX classifier → bucket  (~10ms, no network)
            Rule-based       → time, date, reminder flag

        Slow path  (classifier not ready, or confidence < threshold):
            Gemini full parse → all fields  (skipped when no_llm_fallback=True)
        """
        if _is_command(content):
            result = _default_result()
            result["_is_command"] = True
            return result

        if _is_trivial(content):
            return _default_result()

        now_local = datetime.now(ZoneInfo(user_timezone or "Asia/Kolkata"))
        today     = now_local.strftime("%Y-%m-%d")
        tomorrow  = (now_local + timedelta(days=1)).strftime("%Y-%m-%d")
        now_str   = now_local.strftime("%Y-%m-%d %H:%M (%A)")
        day_map   = {
            (now_local + timedelta(days=i+1)).strftime("%A").lower():
            (now_local + timedelta(days=i+1)).strftime("%Y-%m-%d")
            for i in range(7)
        }

        # ── Forced bucket: bypass classifier and LLM entirely ────
        if force_bucket:
            print(f"[intent] forced bucket → {force_bucket} for: {content[:60]}")
            result = self._build_from_bucket(
                content, force_bucket, today, tomorrow, day_map, check_query=False,
            )
            result["_forced_bucket"] = True
            return result

        # ── Fast path: local ONNX classifier ─────────────────────
        from services.classifier_service import classifier_service, CONF_THRESHOLD
        bucket, confidence = "", 0.0
        if classifier_service.is_ready:
            bucket, confidence = classifier_service.classify(content)

        # ── List format pre-check (~microseconds, no network) ────
        # Run regex list detection regardless of classifier confidence.
        # If a list structure is detected, use the classifier bucket to
        # determine semantics (To-Do task group vs Remember collection).
        from services.list_service import ListService as _LS
        _ls_tmp = _LS(None)
        list_result = _ls_tmp._regex_detect(content)
        if list_result and list_result.get("intent") == "create_or_add" and list_result.get("items"):
            if confidence >= CONF_THRESHOLD and bucket:
                print(f"[intent] list+classifier → {bucket} ({confidence:.2f}) for: {content[:60]}")
                result = self._build_list_result(
                    content, list_result, bucket, today, tomorrow, day_map, check_query,
                )
                result["_classifier_confidence"] = confidence
                return result
            # Low confidence — fall through to LLM (handles save_as_list with bucket already)
            # Unless no_llm_fallback — then use rule-based bucket for the list
            if no_llm_fallback:
                fallback_bucket = _infer_bucket_from_rules(content)
                print(f"[intent] list+no_llm → {fallback_bucket} (rules) for: {content[:60]}")
                result = self._build_list_result(
                    content, list_result, fallback_bucket, today, tomorrow, day_map, check_query=False,
                )
                return result

        if confidence >= CONF_THRESHOLD and bucket:
            # Events always use the LLM — calendar dates must be exact and natural-language
            # date expressions ("two weeks from now", "end of next month", etc.) can't be
            # reliably covered by regex. Latency tradeoff is acceptable for low-volume Events.
            if bucket != "Events":
                print(f"[intent] classifier → {bucket} ({confidence:.2f}) for: {content[:60]}")
                result = self._build_from_bucket(
                    content, bucket, today, tomorrow, day_map, check_query,
                )
                result["_classifier_confidence"] = confidence
                return result
            print(f"[intent] classifier→Events ({confidence:.2f}), deferring to LLM for date accuracy: {content[:60]}")

        # ── Slow path: Gemini full parse (or rule-based if no_llm_fallback) ─
        if no_llm_fallback:
            fallback_bucket = _infer_bucket_from_rules(content)
            print(f"[intent] no_llm_fallback → {fallback_bucket} (rules) for: {content[:60]}")
            return self._build_from_bucket(
                content, fallback_bucket, today, tomorrow, day_map, check_query=False,
            )

        try:
            return await self._llm_parse(
                content, user_name, now_str, today, tomorrow, day_map,
                check_query=check_query,
            )
        except Exception as e:
            print(f"[intent] LLM parse failed: {e}")
            # Events: classifier was confident about the bucket — use regex date extraction
            # as fallback rather than dropping the capture entirely.
            if bucket == "Events" and confidence >= CONF_THRESHOLD:
                print(f"[intent] Events LLM fallback → regex for: {content[:60]}")
                result = self._build_from_bucket(
                    content, "Events", today, tomorrow, day_map, check_query=False,
                )
                result["_classifier_confidence"] = confidence
                return result
            return _default_result()

    # ── Word-form number helpers (shared by fast path) ────────────
    _WORD_TO_NUM = {
        "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8",
        "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
    }
    # Meal/time-of-day context → PM preference
    _PM_CONTEXT  = {"lunch", "afternoon", "evening", "dinner", "night", "tonight", "supper"}
    _AM_CONTEXT  = {"breakfast", "morning", "dawn"}

    @classmethod
    def _normalize_time_words(cls, text: str) -> str:
        """Convert spoken number words to digits inside time expressions.

        "at two" → "at 2", "three pm" → "3 pm", "at five thirty" → "at 5:30"
        """
        import re
        t = text
        half_past = re.search(
            r'\b(?:at\s+)?half\s+past\s+(' + '|'.join(cls._WORD_TO_NUM) + r')\b', t, re.IGNORECASE
        )
        if half_past:
            word = half_past.group(1).lower()
            t = t[:half_past.start()] + f"at {cls._WORD_TO_NUM[word]}:30" + t[half_past.end():]

        quarter_to = re.search(
            r'\b(?:at\s+)?quarter\s+to\s+(' + '|'.join(cls._WORD_TO_NUM) + r')\b', t, re.IGNORECASE
        )
        if quarter_to:
            word  = quarter_to.group(1).lower()
            h     = int(cls._WORD_TO_NUM[word])
            prev  = h - 1 if h > 1 else 12
            t = t[:quarter_to.start()] + f"at {prev}:45" + t[quarter_to.end():]

        for word, digit in cls._WORD_TO_NUM.items():
            # "at/by/around <word> [thirty]" → "at/by/around <digit>[:30]"
            t = re.sub(
                rf'(?<!\w)(?:at|by|around)\s+{word}\s+thirty\b',
                f'at {digit}:30', t, flags=re.IGNORECASE
            )
            t = re.sub(
                rf'(?<!\w)(?:at|by|around)\s+{word}\b',
                f'at {digit}', t, flags=re.IGNORECASE
            )
            # "<word> am/pm" standalone
            t = re.sub(rf'\b{word}\s+(am|pm)\b', f'{digit} \\1', t, flags=re.IGNORECASE)
        return t

    @classmethod
    def _infer_meridiem(cls, text: str, hour: int) -> bool:
        """Return True (PM) or False (AM) when no am/pm marker is present."""
        lc = text.lower()
        if any(s in lc for s in cls._PM_CONTEXT):
            return True
        if any(s in lc for s in cls._AM_CONTEXT):
            return False
        # Heuristic: unspecified hours 1–6 are almost always PM in everyday speech
        return 1 <= hour <= 6

    def _build_from_bucket(
        self,
        content: str,
        bucket: str,
        today: str,
        tomorrow: str,
        day_map: Dict,
        check_query: bool,
    ) -> Dict:
        """
        Build the full actions dict from a classifier bucket + rule-based signals.
        This replaces the LLM for the common case.
        """
        import re
        from datetime import datetime

        result  = _default_result()
        actions = result["actions"]
        lc      = self._normalize_time_words(content.lower())

        # Normalise non-standard time notations before the am/pm regex runs,
        # so that the main pattern always sees clean "H:MM am/pm" strings.
        #
        # 1. Decimal: "11.0 am" → "11:00 am", "3.30 pm" → "3:30 pm"
        #    Without this, \b before the trailing digit (e.g. "0" in "11.0")
        #    causes the am/pm regex to match "0 am" → midnight (00:00).
        lc = re.sub(
            r'\b(\d{1,2})\.(\d{1,2})\s*(am|pm)\b',
            lambda m: f"{m.group(1)}:{int(m.group(2)):02d} {m.group(3)}",
            lc
        )
        # 2. Space-separated: "3 07 pm" → "3:07 pm", "10 45 am" → "10:45 am"
        #    Without this, the regex grabs only the minute digits (e.g. "07 pm"
        #    → 07+12=19:00) and loses the hour entirely, or produces impossible
        #    values like "30 pm" → 42:00.
        lc = re.sub(
            r'\b(\d{1,2})\s+(\d{2})\s*(am|pm)\b',
            lambda m: f"{m.group(1)}:{m.group(2)} {m.group(3)}",
            lc
        )

        # ── Query detection (rule-based, only when check_query=True) ─
        if check_query and "?" in content:
            query_signals = {"find", "search", "show", "what did", "where is",
                             "recall", "do i have", "show me"}
            if any(sig in lc for sig in query_signals):
                actions["is_query"] = True
                result["query"] = {"query_text": content, "date_hint": None, "list_name": None}
                result["essence"] = content[:100]
                return result

        # ── Map bucket → primary action ───────────────────────────
        bucket_to_action = {
            "To-Do":    "save_as_todo",
            "Events":   "save_as_event",
            "Remember": "save_as_note",
            "Ideas":    "save_as_idea",
            "Track":    "save_as_track",
            "Random":   "save_as_note",
        }
        primary_action = bucket_to_action.get(bucket, "save_as_note")
        actions[primary_action] = True

        # ── Time / date extraction (rule-based) ───────────────────
        time_str: Optional[str] = None
        date_str: Optional[str] = None

        # Specific time with explicit am/pm: "10pm", "10:30 am"
        m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", lc)
        if m:
            h    = int(m.group(1))
            mins = int(m.group(2) or 0)
            if m.group(3) == "pm" and h != 12:
                h += 12
            elif m.group(3) == "am" and h == 12:
                h = 0
            time_str = f"{h:02d}:{mins:02d}"
        elif "noon" in lc:
            time_str = "12:00"
        elif "midnight" in lc:
            time_str = "00:00"
        else:
            # "at N" or "at N:MM" without explicit am/pm — infer meridiem from context
            m_bare = re.search(r"\bat\s+(\d{1,2})(?::(\d{2}))?\b", lc)
            if m_bare:
                h    = int(m_bare.group(1))
                mins = int(m_bare.group(2) or 0)
                if self._infer_meridiem(lc, h) and h != 12:
                    h += 12
                time_str = f"{h:02d}:{mins:02d}"
            else:
                # Relative time: "in 30 minutes" / "in 2 hours" / "after 45 mins"
                # Compute absolute time from now so reminder fires at the right moment.
                rel_m = re.search(r"\b(?:in|after)\s+(\d+)\s*(?:min|mins|minute|minutes)\b", lc)
                rel_h = re.search(r"\b(?:in|after)\s+(\d+)\s*(?:hour|hours|hr|hrs)\b", lc)
                if rel_m or rel_h:
                    from datetime import datetime as _dt
                    from zoneinfo import ZoneInfo as _ZI
                    _now = _dt.now(_ZI("Asia/Kolkata"))
                    if rel_m:
                        _target = _now + timedelta(minutes=int(rel_m.group(1)))
                    else:
                        _target = _now + timedelta(hours=int(rel_h.group(1)))
                    time_str = _target.strftime("%H:%M")
                    date_str = _target.strftime("%Y-%m-%d")  # may cross midnight

        # Date: "today", "tomorrow", day-of-week
        if not date_str:
            if "today" in lc:
                date_str = today
            elif "tonight" in lc or "this evening" in lc:
                date_str = today
                # "tonight" / "this evening" with no explicit time → default 20:00
                if not time_str:
                    time_str = "20:00"
            elif "tomorrow" in lc:
                date_str = tomorrow
            else:
                for day_name, day_date in day_map.items():
                    if day_name in lc:
                        date_str = day_date
                        break

        # Specific calendar date: "17th June", "June 17", "3 July", "12-14th August"
        # Handles formats with or without ordinal suffix and optional whitespace.
        if not date_str:
            _MONTHS = {
                "january": 1, "february": 2, "march": 3, "april": 4,
                "may": 5, "june": 6, "july": 7, "august": 8,
                "september": 9, "october": 10, "november": 11, "december": 12,
                "jan": 1, "feb": 2, "mar": 3, "apr": 4,
                "jun": 6, "jul": 7, "aug": 8,
                "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
            }
            _month_pat = (
                r"january|february|march|april|may|june|july|august|september|"
                r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec"
            )
            # Day then month: "17th June", "12-14th August", "3july"
            m_dm = re.search(
                rf"\b(\d{{1,2}})(?:-\d{{1,2}})?(?:st|nd|rd|th)?\s?({_month_pat})\b", lc
            )
            # Month then day: "June 17", "June 17th", "july3"
            m_md = re.search(
                rf"\b({_month_pat})\s?(\d{{1,2}})(?:st|nd|rd|th)?\b", lc
            )
            _day, _mon = 0, 0
            if m_dm:
                _day = int(m_dm.group(1))
                _mon = _MONTHS.get(m_dm.group(2), 0)
            elif m_md:
                _mon = _MONTHS.get(m_md.group(1), 0)
                _day = int(m_md.group(2))
            if _mon and 1 <= _day <= 31:
                from datetime import date as _date
                today_date = _date.fromisoformat(today)
                try:
                    candidate = _date(today_date.year, _mon, _day)
                    if candidate < today_date:
                        candidate = _date(today_date.year + 1, _mon, _day)
                    date_str = candidate.isoformat()
                except ValueError:
                    pass  # invalid date (e.g. Feb 30) — fall through to today

        if not date_str:
            date_str = today

        # ── Reminder flag ─────────────────────────────────────────
        reminder_kw = {"remind", "reminder", "don't forget", "alert", "notify", "ping"}
        if any(kw in lc for kw in reminder_kw) and time_str:
            actions["set_reminder"] = True
            actions["save_as_todo"] = True
            result["reminder"] = {"due_date": date_str, "time": time_str, "priority": "normal"}

        # Also set reminder when time is explicitly mentioned with a todo
        if time_str and (bucket in ("To-Do", "Events")):
            actions["set_reminder"] = True
            result["reminder"] = {"due_date": date_str, "time": time_str, "priority": "normal"}

        # ── Priority ──────────────────────────────────────────────
        priority = "normal"
        if any(w in lc for w in {"urgent", "asap", "critical", "emergency"}):
            priority = "urgent"
        elif any(w in lc for w in {"important", "must", "definitely", "high priority"}):
            priority = "high"

        # ── Bullet list for To-Do with 2+ items ──────────────────────
        # Named header + bullets → is_list=True (expandable ListTodoRow).
        # Neutral headers (todo/tasks/today/tomorrow/this week) → individual To-Do rows,
        # matching the LLM NEUTRAL HEADER RULE so fast and slow paths behave the same.
        if bucket == "To-Do":
            has_bullets = bool(re.search(r'\n\s*[-*•]|\n\s*\d+[.)]', content))
            if has_bullets:
                from services.list_service import _extract_items_from_content, _BLOCKED_NAME_WORDS
                items = _extract_items_from_content(content)
                if len(items) >= 2:
                    header_m = re.match(r'^([^\n]+)', content)
                    raw_header = (
                        header_m.group(1).strip().rstrip(':').strip()
                        if header_m else ""
                    ) or "Tasks"
                    # Neutral headers → split as individual tasks (mirrors LLM behaviour).
                    # Neutral = ALL non-stopword words are blocked words (e.g. "todo for today").
                    # Named qualifier present (e.g. "Em issues for today", "Office work") → named list.
                    _h_stopwords = {"for", "this", "my", "the", "a", "an", "of", "in", "on"}
                    header_words = set(raw_header.lower().split()) - _h_stopwords
                    is_neutral = not bool(header_words - _BLOCKED_NAME_WORDS)
                    if is_neutral:
                        # Save as separate To-Do tasks, not a named list
                        result["tasks"] = [
                            {"task": item, "due_date": date_str, "time": time_str, "priority": priority}
                            for item in items
                        ]
                        result["priority"] = priority
                        result["essence"]  = content[:100]
                        return result
                    # Named header → named list
                    actions["save_as_list"] = True
                    result["list"] = {
                        "list_name": raw_header,
                        "list_type": "todo",
                        "items":     items,
                        "due_date":  date_str,
                        "bucket":    "To-Do",
                    }
                    result["priority"] = priority
                    result["essence"]  = content[:100]
                    return result

        # ── Build task / note / etc. ──────────────────────────────
        if actions["save_as_todo"]:
            result["tasks"] = [{
                "task":     content,
                "due_date": date_str,
                "time":     time_str,
                "priority": priority,
            }]

        if actions["save_as_event"]:
            result["event"] = {
                "title":    content[:80],
                "due_date": date_str,
                "time":     time_str,
                "people":   [],
            }

        if actions["save_as_note"]:
            result["note"] = {"content": content, "keywords": []}

        if actions["save_as_idea"]:
            result["idea"] = {"content": content, "keywords": []}

        result["priority"] = priority
        result["essence"]  = content[:100]
        return result

    def _build_list_result(
        self,
        content: str,
        list_result: Dict,
        bucket: str,
        today: str,
        tomorrow: str,
        day_map: Dict,
        check_query: bool,
    ) -> Dict:
        """
        Build an actions dict for a list-formatted capture.
        The classifier bucket determines list semantics:
          To-Do   → task group (parent with sub-todos, mutable via "add X to Y")
          Remember → named collection (mutable via "add X to Y")
          Track   → each item is a separate log entry (no persistent list)
          Others  → stored as is_list=True with the appropriate bucket action
        """
        result  = _default_result()
        actions = result["actions"]

        if bucket == "To-Do":
            actions["save_as_todo"] = True
            actions["save_as_list"] = True
        elif bucket == "Remember":
            actions["save_as_note"] = True
            actions["save_as_list"] = True
        elif bucket == "Track":
            actions["save_as_track"] = True
            # Track items are individual log entries — no persistent named list
        else:
            primary = {
                "Events": "save_as_event",
                "Ideas":  "save_as_idea",
                "Random": "save_as_note",
            }.get(bucket, "save_as_note")
            actions[primary] = True
            actions["save_as_list"] = True

        result["list"] = {
            "list_name": list_result["list_name"],
            "list_type": list_result["list_type"],
            "items":     list_result["items"],
            "due_date":  None,
            "bucket":    bucket,
        }
        result["essence"] = content[:100]
        return result

    async def _llm_parse(
        self,
        content: str,
        user_name: str,
        now_str: str,
        today: str,
        tomorrow: str,
        day_map: Dict,
        check_query: bool = True,
    ) -> Dict:

        header = (
            f"You are parsing a message from {user_name} for their personal second brain — an app where people dump fragments of thought so they can find them later.\n\n"
            f"NOW: {now_str}\n"
            f"TODAY: {today}\n"
            f"TOMORROW: {tomorrow}\n"
            f"NEXT 7 DAYS: {day_map}\n\n"
            f"MESSAGE:\n\"\"\"\n{content}\n\"\"\"\n\n"
            "TASK: Determine ALL save actions this message should trigger.\n\n"
            "BEFORE CLASSIFYING — ask one question: what does the user want to do with this later?\n"
            "  → Look it up / recall it        = save_as_note\n"
            "  → Act on it / check it off      = save_as_todo\n"
            "  → Track it over time            = save_as_track\n"
            "Users often dump bare words, document names, or facts with zero elaboration. That is deliberate — they are filing a reference, not creating a task. Default to save_as_note for anything that lacks a clear action signal.\n\n"
        )

        if check_query:
            action_flags = (
                "ACTION FLAGS:\n"
                "  save_as_todo  — message contains one or more tasks/things to do\n"
                "  save_as_event — message is a scheduled appointment/meeting with a specific time\n"
                "  save_as_note  — message is a fact/info to remember (credential, location, info)\n"
                "  save_as_idea  — message is a creative thought/insight/concept\n"
                "  save_as_track — message logs health/habit data (weight, steps, mood, sleep)\n"
                "  save_as_list  — message is a named list (shopping, bag, packing) with items\n"
                "  set_reminder  — message has a specific time → should create a reminder\n"
                "  is_query      — user wants to retrieve/search something they saved before\n\n"
                "is_query=true ONLY for: questions ending with '?', or explicit search phrases\n"
                "  ('show me', 'find', 'search', 'what did I save', 'recall', 'where is', 'do I have').\n"
                "  Imperative statements are NEVER is_query — they are save_as_todo.\n\n"
            )
            json_schema = (
                "Return ONLY this JSON (no markdown):\n"
                "{\n"
                '  "actions": {"save_as_todo":bool,"save_as_event":bool,"save_as_note":bool,'
                '"save_as_idea":bool,"save_as_track":bool,"save_as_list":bool,"set_reminder":bool,"is_query":bool},\n'
                '  "tasks": [{"task":"...","due_date":"YYYY-MM-DD","time":"HH:MM|null","priority":"normal|high|urgent"}],\n'
                '  "reminder": {"due_date":"YYYY-MM-DD","time":"HH:MM","priority":"normal"} | null,\n'
                '  "event": {"title":"...","due_date":"YYYY-MM-DD","time":"HH:MM|null","people":[]} | null,\n'
                '  "list": {"list_name":"...","list_type":"shopping|bag|packing|reading|watching|custom","items":[],"due_date":"YYYY-MM-DD|null"} | null,\n'
                '  "track": {"logs":[{"metric":"...","value":"...","unit":"..."}]} | null,\n'
                '  "note": {"content":"...","keywords":[]} | null,\n'
                '  "idea": {"content":"...","keywords":[]} | null,\n'
                '  "query": {"query_text":"...","date_hint":"today|tomorrow|this_week|null","list_name":"...|null"} | null,\n'
                '  "people":[],"priority":"normal|high|urgent","essence":"one sentence summary"\n'
                "}\n\n"
            )
        else:
            # Save-only mode: no is_query, shorter prompt, faster
            action_flags = (
                "ACTION FLAGS (classify by what the user intends to do with this later):\n"
                "  save_as_todo  — user intends to PERFORM an action (something to check off)\n"
                "  save_as_event — scheduled appointment/meeting with a specific time\n"
                "  save_as_note  — user intends to RECALL information (something to look up)\n"
                "  save_as_idea  — creative thought/insight/concept\n"
                "  save_as_track — logs health/habit data (weight, steps, mood, sleep)\n"
                "  save_as_list  — named list (shopping, bag, packing) with items\n"
                "  set_reminder  — has a specific time → create a reminder\n"
                "Bare nouns, document names, and facts with no action signal → save_as_note.\n\n"
            )
            json_schema = (
                "Return ONLY this JSON (no markdown):\n"
                "{\n"
                '  "actions": {"save_as_todo":bool,"save_as_event":bool,"save_as_note":bool,'
                '"save_as_idea":bool,"save_as_track":bool,"save_as_list":bool,"set_reminder":bool,"is_query":false},\n'
                '  "tasks": [{"task":"...","due_date":"YYYY-MM-DD","time":"HH:MM|null","priority":"normal|high|urgent"}],\n'
                '  "reminder": {"due_date":"YYYY-MM-DD","time":"HH:MM","priority":"normal"} | null,\n'
                '  "event": {"title":"...","due_date":"YYYY-MM-DD","time":"HH:MM|null","people":[]} | null,\n'
                '  "list": {"list_name":"...","list_type":"shopping|bag|packing|reading|watching|custom","items":[],"due_date":"YYYY-MM-DD|null"} | null,\n'
                '  "track": {"logs":[{"metric":"...","value":"...","unit":"..."}]} | null,\n'
                '  "note": {"content":"...","keywords":[]} | null,\n'
                '  "idea": {"content":"...","keywords":[]} | null,\n'
                '  "people":[],"priority":"normal|high|urgent","essence":"one sentence summary"\n'
                "}\n\n"
            )

        rules = (
            "KEY RULES:\n"
            "  TODO vs NOTE — the only question that matters is intent:\n"
            "    save_as_todo  = user intends to PERFORM an action (there is something to check off)\n"
            "    save_as_note  = user intends to RECALL information (there is something to look up)\n"
            "  A bare noun, document name, place, person detail, or fact → save_as_note.\n"
            "  Require a genuine action signal for save_as_todo: an explicit or strongly implied verb\n"
            "  ('call', 'buy', 'submit', 'book', 'pay', 'fix', 'check', 'need to', 'have to') OR\n"
            "  time pressure ('by Friday', 'urgent', 'before 5pm'). Without one → save_as_note.\n\n"
            "  - A reminder IS ALWAYS also a todo → save_as_todo AND set_reminder = true\n"
            "  - save_as_event = true for scheduled appointments/meetings; save_as_todo = false for pure events\n"
            "  - set_reminder = true ONLY when a specific time is mentioned\n"
            "  - Time without explicit date → assume TODAY\n"
            "  - Any task/actionable without a date → due_date = TODAY\n"
            "  - Extract ALL tasks even from unstructured prose\n"
            "  - Priority: urgent/asap/critical/important → high, else normal\n\n"
            "  NAMED HEADER RULE:\n"
            "  Named header (project/brand/place/person) + bullet items → save_as_list=true\n"
            "  Named qualifier + 'tasks/todos' + items → save_as_list=true (the qualifier makes it named)\n"
            "  Neutral header (ONLY generic words: Todo, Tasks, Today, Tomorrow, This week) + bullet items → save_as_todo=true (NOT list)\n"
            "  When a list has a date (e.g. 'for tomorrow', 'by Friday'), extract it as due_date on the list object.\n\n"
            "  NAMED: 'Extended minds changes', 'Japan trip', 'Grocery', 'Dmart', 'Client ABC', 'Office tasks', 'Office tasks for tomorrow'\n"
            "  NEUTRAL: 'Todo', 'Tasks', 'Today', 'Todo for today', 'Tomorrow', 'This week'\n\n"
        )

        examples = (
            "EXAMPLES:\n\n"
            f'M: "remind me to call mom at 10pm" → save_as_todo=true, set_reminder=true\n'
            f'  tasks:[{{"task":"call mom","due_date":"{today}","time":"22:00","priority":"normal"}}]\n\n'

            f'M: "dentist appointment Friday 3pm" → save_as_todo=true, save_as_event=true, set_reminder=true\n'
            f'  tasks:[{{"task":"dentist","due_date":"{day_map.get("friday", tomorrow)}","time":"15:00","priority":"normal"}}]\n\n'

            f'M: "marriage certificate" → save_as_note=true (user is filing a reference, not creating a task)\n'
            f'  note:{{"content":"marriage certificate","keywords":["marriage","certificate","document"]}}\n\n'

            f'M: "passport" → save_as_note=true\n'
            f'  note:{{"content":"passport","keywords":["passport","document"]}}\n\n'

            f'M: "submit passport application" → save_as_todo=true (explicit action verb)\n'
            f'  tasks:[{{"task":"submit passport application","due_date":"{today}","time":null,"priority":"normal"}}]\n\n'

            f'M: "Todo for today:\\n- Check apple dev\\n- Pack for trip" → save_as_todo=true (NOT save_as_list, purely generic header)\n'
            f'  tasks:[{{"task":"Check apple dev","due_date":"{today}","time":null,"priority":"normal"}},{{"task":"Pack for trip","due_date":"{today}","time":null,"priority":"normal"}}]\n\n'

            f'M: "New todo list for tomorrow:\\n- Work on meraki\\n- Follow up with Aditi" → save_as_todo=true (NOT save_as_list, purely generic header)\n'
            f'  tasks:[{{"task":"Work on meraki","due_date":"{tomorrow}","time":null,"priority":"normal"}},{{"task":"Follow up with Aditi","due_date":"{tomorrow}","time":null,"priority":"normal"}}]\n\n'

            f'M: "Office tasks for tomorrow:\\n- Check apple dev\\n- Email Aditi" → save_as_list=true (\'Office\' is a named qualifier)\n'
            f'  list:{{"list_name":"Office Tasks","list_type":"custom","items":["Check apple dev","Email Aditi"],"due_date":"{tomorrow}"}}\n\n'

            f'M: "Client XYZ todos:\\n- Send proposal\\n- Follow up on call" → save_as_list=true (\'Client XYZ\' is a named qualifier)\n'
            f'  list:{{"list_name":"Client XYZ Tasks","list_type":"custom","items":["Send proposal","Follow up on call"],"due_date":null}}\n\n'

            f'M: "dmart shopping:\\n- milk\\n- eggs" → save_as_list=true\n'
            f'  list:{{"list_name":"Dmart Shopping List","list_type":"shopping","items":["milk","eggs"]}}\n\n'

            f'M: "Japan trip:\\n- book flights\\n- get visa" → save_as_list=true, save_as_todo=true\n'
            f'  list:{{"list_name":"Japan Trip List","list_type":"packing","items":["book flights","get visa"]}}\n'
            f'  tasks:[{{"task":"book flights","due_date":null,"time":null,"priority":"normal"}},{{"task":"get visa","due_date":null,"time":null,"priority":"normal"}}]\n\n'

            f'M: "weight 74kg, slept 7h" → save_as_track=true\n'
            f'  track:{{"logs":[{{"metric":"weight","value":"74","unit":"kg"}},{{"metric":"sleep","value":"7","unit":"hours"}}]}}\n\n'

            f'M: "wifi password is airtel123" → save_as_note=true\n'
            f'  note:{{"content":"wifi password is airtel123","keywords":["wifi","password"]}}\n\n'
        )

        if check_query:
            examples += (
                f'M: "what did I save about Japan" → is_query=true\n'
                f'  query:{{"query_text":"Japan","date_hint":null,"list_name":null}}\n\n'
                f'M: "show me my grocery list" → is_query=true\n'
                f'  query:{{"query_text":"grocery list","date_hint":null,"list_name":"Grocery List"}}\n\n'
                f'M: "find my wifi password" → is_query=true\n'
                f'  query:{{"query_text":"wifi password","date_hint":null,"list_name":null}}\n\n'
            )

        prompt = header + action_flags + json_schema + rules + examples
        response = await self.fast.chat(prompt, max_tokens=600)
        return self._validate(response)

    def _validate(self, raw: Dict) -> Dict:
        """Validate and sanitise LLM output. Fill missing fields with safe defaults."""
        result = _default_result()

        # Actions
        actions_raw = raw.get("actions", {})
        if isinstance(actions_raw, dict):
            for key in result["actions"]:
                val = actions_raw.get(key, False)
                result["actions"][key] = bool(val)

        # Enforce action coherence
        # If set_reminder=true, save_as_todo must also be true
        if result["actions"]["set_reminder"]:
            result["actions"]["save_as_todo"] = True
        # If save_as_event=true, save_as_todo must also be true
        if result["actions"]["save_as_event"]:
            result["actions"]["save_as_todo"] = True

        # Tasks
        tasks_raw = raw.get("tasks", [])
        if isinstance(tasks_raw, list):
            for t in tasks_raw:
                if not isinstance(t, dict):
                    continue
                task_text = str(t.get("task", "")).strip()
                if not task_text:
                    continue
                due_date = t.get("due_date")
                if not due_date or not _DATE_RE.match(str(due_date)):
                    due_date = _today_str()
                result["tasks"].append({
                    "task":     task_text,
                    "due_date": due_date,
                    "time":     t.get("time"),
                    "priority": t.get("priority", "normal")
                    if t.get("priority") in ("normal", "high", "urgent") else "normal",
                })

        # Reminder
        rem = raw.get("reminder")
        if isinstance(rem, dict) and result["actions"]["set_reminder"]:
            due = rem.get("due_date")
            if not due or not _DATE_RE.match(str(due)):
                due = _today_str()
            result["reminder"] = {
                "due_date": due,
                "time":     rem.get("time"),
                "priority": rem.get("priority", "normal"),
            }

        # Event
        evt = raw.get("event")
        if isinstance(evt, dict) and result["actions"]["save_as_event"]:
            due = evt.get("due_date")
            if not due or not _DATE_RE.match(str(due)):
                due = _today_str()
            result["event"] = {
                "title":    str(evt.get("title", ""))[:80],
                "due_date": due,
                "time":     evt.get("time"),
                "people":   evt.get("people", []) if isinstance(evt.get("people"), list) else [],
            }

        # List
        lst = raw.get("list")
        if isinstance(lst, dict) and result["actions"]["save_as_list"]:
            items = lst.get("items", [])
            if isinstance(items, list) and items:
                list_due = lst.get("due_date")
                if list_due and not _DATE_RE.match(str(list_due)):
                    list_due = None
                result["list"] = {
                    "list_name": str(lst.get("list_name", "My List")),
                    "list_type": lst.get("list_type", "custom")
                    if lst.get("list_type") in ("shopping","bag","packing","reading","watching","custom")
                    else "custom",
                    "items":    [str(i).strip() for i in items if str(i).strip()],
                    "due_date": list_due,
                }

        # Track
        trk = raw.get("track")
        if isinstance(trk, dict) and result["actions"]["save_as_track"]:
            logs = trk.get("logs", [])
            if isinstance(logs, list):
                result["track"] = {"logs": [l for l in logs if isinstance(l, dict)]}

        # Note
        note = raw.get("note")
        if isinstance(note, dict) and result["actions"]["save_as_note"]:
            result["note"] = {
                "content":  str(note.get("content", ""))[:500],
                "keywords": note.get("keywords", []) if isinstance(note.get("keywords"), list) else [],
            }

        # Idea
        idea = raw.get("idea")
        if isinstance(idea, dict) and result["actions"]["save_as_idea"]:
            result["idea"] = {
                "content":  str(idea.get("content", ""))[:500],
                "keywords": idea.get("keywords", []) if isinstance(idea.get("keywords"), list) else [],
            }

        # Query
        qry = raw.get("query")
        if isinstance(qry, dict) and result["actions"]["is_query"]:
            result["query"] = {
                "query_text": str(qry.get("query_text", ""))[:200],
                "date_hint":  qry.get("date_hint") if qry.get("date_hint") in
                              ("today", "tomorrow", "this_week", None) else None,
                "list_name":  qry.get("list_name"),
            }

        # People, priority, essence
        result["people"]   = raw.get("people", []) if isinstance(raw.get("people"), list) else []
        result["priority"] = raw.get("priority", "normal") \
            if raw.get("priority") in ("normal", "high", "urgent") else "normal"
        result["essence"]  = str(raw.get("essence", ""))[:200]

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (used in validation and rule-based fallback)
# ─────────────────────────────────────────────────────────────────────────────

import re
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_TRACK_SIGNALS = {"kg", "km", "mile", "steps", "calories", "kcal", "mood", "slept", "weight", "bp", "sugar", "water"}
_ACTION_VERBS = {
    "call", "email", "buy", "get", "pick", "order", "send", "submit", "upload",
    "pay", "fix", "update", "review", "check", "book", "schedule", "remind",
    "need to", "have to", "must", "should", "write", "draft", "create", "build",
    "read", "do", "take", "make", "go", "add", "remove", "print", "find",
}

def _infer_bucket_from_rules(content: str) -> str:
    """Rule-based bucket when classifier is unavailable and LLM is disabled (group captures)."""
    # Strip @mentions before analysis
    clean = re.sub(r'@\w+', '', content).strip().lower()

    # Track: numeric health/habit signals
    if any(s in clean for s in _TRACK_SIGNALS) and re.search(r"\d", clean):
        return "Track"

    # Ideas
    if re.search(r"\b(idea|concept|what if|imagine|startup|feature|suggestion)\b", clean):
        return "Ideas"

    # Action verbs or reminder keywords → To-Do
    if (
        any(re.search(rf'(?<!\w){re.escape(v)}(?!\w)', clean) for v in _ACTION_VERBS)
        or re.search(r"\b(remind|reminder|don.t forget|notify|alert)\b", clean)
        or re.search(r"\b(\d{1,2}(am|pm)|\d{1,2}:\d{2}|noon|midnight|today|tomorrow)\b", clean)
        or re.search(r"\b(in|after)\s+\d+\s*(min|mins|minute|minutes|hour|hours|hr|hrs)\b", clean)
    ):
        return "To-Do"

    # Substantive content without action → Remember
    if len(clean) > 15:
        return "Remember"
    return "Random"

def _today_str() -> str:
    from datetime import datetime
    return datetime.utcnow().strftime("%Y-%m-%d")


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_intent_service: Optional[IntentService] = None

def get_intent_service(cerebras_client: CerebrasClient) -> IntentService:
    global _intent_service
    if _intent_service is None:
        _intent_service = IntentService(cerebras_client)
    return _intent_service
