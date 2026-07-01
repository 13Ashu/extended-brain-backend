"""
Intelligent Message Processor — v3
────────────────────────────────────────────────────────────────────────────
KEY CHANGES over v2:
  • "List" bucket — named lists (shopping, bag, packing) routed to list_service
  • Priority detection — urgent/high affects reminder re-fire frequency
  • List intent detected BEFORE todo batch splitting
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
import pytz

# Date indicator words — if present, skip regex fast path so LLM can extract due_date for lists
_LIST_DATE_RE = re.compile(
    r'\b(today|tomorrow|tonight|this week|next week|monday|tuesday|wednesday|thursday|friday|saturday|sunday|by \w+|in \d+ days?)\b',
    re.IGNORECASE,
)

_IST = pytz.timezone("Asia/Kolkata")
from typing import Dict, List, Optional, Tuple

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from database import Category, Message, User
from cerebras_client import CerebrasClient
from models import MessageType


# ─────────────────────────────────────────────────────────────────────────────
# INTENT BUCKETS
# ─────────────────────────────────────────────────────────────────────────────

INTENT_BUCKETS: Dict[str, str] = {
    "Remember": (
        "Recall a fact, location, credential, or piece of information for later. "
        "Examples: parked at C3, nailcutter in 3rd drawer, wifi password 1234."
    ),
    "To-Do": (
        "A task or action item the user must act on. "
        "Examples: buy milk, call dentist, submit report by Friday, call Kailash at noon."
    ),
    "Ideas": (
        "A new idea, thought, concept, or creative spark. "
        "Examples: startup idea, feature suggestion, book concept."
    ),
    "Track": (
        "Log or monitor something — health, habits, progress, mood. "
        "Examples: weight 74kg, ran 5km, mood anxious, slept 6 hours."
    ),
    "Events": (
        "A time-anchored event or appointment with a specific date/time. "
        "Examples: dentist Friday 3pm, flight to Delhi on 20th."
    ),
    "List": (
        "A named collection of items — shopping list, packing list, bag list, reading list. "
        "Examples: shopping list with groceries, packing list for trip, bag list for exam."
    ),
    "Random": (
        "Casual, venting, conversational, or unclear. "
        "Examples: heyy, lol okay, ugh today was rough."
    ),
}

BUCKET_NAMES = list(INTENT_BUCKETS.keys())


def _implicit_search_keywords(message_type: str, media_url: Optional[str]) -> Optional[str]:
    """
    Compute space-separated implicit search keywords baked into tags at save time.
    These let users type natural words ("insta", "image", "links") and instantly
    find captures via ILIKE — both server-side and on the iOS local search index
    (which has no access to media_url or message_type at query time).
    """
    parts: List[str] = []
    if media_url and media_url.startswith("http"):
        url = media_url.lower()
        parts += ["link", "links", "url", "website"]
        if "instagram" in url:
            parts += ["instagram", "insta"]
            if "/reel" in url:
                parts += ["reel", "reels"]
        elif "youtu" in url:
            parts += ["youtube", "yt", "video"]
        elif "twitter" in url or "x.com" in url:
            parts += ["twitter", "tweet"]
        elif "linkedin" in url:
            parts.append("linkedin")
        elif "reddit" in url:
            parts.append("reddit")
        elif "github" in url:
            parts.append("github")
    elif message_type == "image":
        parts += ["image", "photo", "pic", "picture"]
    # documents: filename (with .pdf / .docx extension) is already folded into content
    # at extraction time, so "pdf" / the filename are already ILIKE-searchable there.
    return " ".join(parts) if parts else None

ACTION_VERBS = {
    "call", "email", "message", "text", "ping", "contact",
    "buy", "get", "pick", "order", "purchase",
    "send", "submit", "upload", "share", "forward",
    "remind", "schedule", "book", "reserve", "register",
    "pay", "transfer", "deposit",
    "fix", "update", "review", "check", "verify",
    "meet", "attend", "join", "visit",
    "write", "draft", "prepare", "create", "build",
    "read", "watch", "listen", "study", "learn",
    "clean", "wash", "organize", "sort",
    "see", "do", "take", "make", "go", "keep", "put",
    "run", "start", "find", "add", "remove", "print",
    "need to", "have to", "must", "should",
}

TRACK_SIGNALS = {
    "kg", "km", "mile", "steps", "calories", "kcal",
    "hours", "mins", "mood", "sleep", "slept", "ran",
    "walked", "weight", "bp", "sugar", "water", "drank",
}

REMINDER_KEYWORDS = {"remind", "reminder", "don't forget", "alert", "notify", "ping"}

PRIORITY_HIGH_SIGNALS = {
    "urgent", "asap", "critical", "important", "must", "definitely",
    "don't forget", "cannot miss", "high priority",
}

PRIORITY_URGENT_SIGNALS = {
    "emergency", "immediately", "right now", "super urgent", "very urgent",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_WORD_TO_NUM = {
    "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
}
_PM_CONTEXT = {"lunch", "afternoon", "evening", "dinner", "night", "tonight", "supper"}
_AM_CONTEXT = {"breakfast", "morning", "dawn"}


def _normalize_time_words(text: str) -> str:
    """Convert spoken number words to digits inside time expressions."""
    t = text
    half_past = re.search(
        r'\b(?:at\s+)?half\s+past\s+(' + '|'.join(_WORD_TO_NUM) + r')\b', t, re.IGNORECASE
    )
    if half_past:
        word = half_past.group(1).lower()
        t = t[:half_past.start()] + f"at {_WORD_TO_NUM[word]}:30" + t[half_past.end():]

    quarter_to = re.search(
        r'\b(?:at\s+)?quarter\s+to\s+(' + '|'.join(_WORD_TO_NUM) + r')\b', t, re.IGNORECASE
    )
    if quarter_to:
        word = quarter_to.group(1).lower()
        h    = int(_WORD_TO_NUM[word])
        prev = h - 1 if h > 1 else 12
        t = t[:quarter_to.start()] + f"at {prev}:45" + t[quarter_to.end():]

    for word, digit in _WORD_TO_NUM.items():
        t = re.sub(
            rf'(?<!\w)(?:at|by|around)\s+{word}\s+thirty\b',
            f'at {digit}:30', t, flags=re.IGNORECASE
        )
        t = re.sub(
            rf'(?<!\w)(?:at|by|around)\s+{word}\b',
            f'at {digit}', t, flags=re.IGNORECASE
        )
        t = re.sub(rf'\b{word}\s+(am|pm)\b', f'{digit} \\1', t, flags=re.IGNORECASE)
    return t


def _infer_meridiem(text: str, hour: int) -> bool:
    """Return True (PM) when no am/pm marker is present, based on context."""
    lc = text.lower()
    if any(s in lc for s in _PM_CONTEXT):
        return True
    if any(s in lc for s in _AM_CONTEXT):
        return False
    return 1 <= hour <= 6


def _today_str() -> str:
    return datetime.now(_IST).strftime("%Y-%m-%d")

def _ist_now() -> datetime:
    # DB stores UTC naive; _today_str() handles IST date logic separately
    return datetime.utcnow()


def _detect_priority(content: str) -> str:
    lc = content.lower()
    if any(sig in lc for sig in PRIORITY_URGENT_SIGNALS):
        return "urgent"
    if any(sig in lc for sig in PRIORITY_HIGH_SIGNALS):
        return "high"
    return "normal"


def _sniff_buckets_fast(content: str) -> List[str]:
    lc = _normalize_time_words(content.lower())

    # Reminder override — explicit reminder request always saves as To-Do + Events if timed
    if any(kw in lc for kw in REMINDER_KEYWORDS):
        buckets = ["To-Do"]
        if re.search(
            r"\b(today|tomorrow|tonight|morning|afternoon|evening|noon|midnight"
            r"|\d{1,2}(am|pm)|\d{1,2}:\d{2}"
            r"|in\s+\d+\s*(min|mins|minute|minutes|hour|hours)"
            r"|after\s+\d+\s*(min|mins|minute|minutes|hour|hours)"
            r"|at\s+\d{1,2})\b", lc
        ):
            buckets.append("Events")
        return buckets

    # Short/greeting
    if len(lc.strip()) < 6 or lc.strip() in {"hi", "hey", "heyy", "ok", "okay", "lol", "test"}:
        return ["Random"]

    buckets: List[str] = []

    # Track
    if any(sig in lc for sig in TRACK_SIGNALS) and re.search(r"\d", lc):
        buckets.append("Track")

    # To-Do
    if any(lc.startswith(v) or f" {v} " in lc for v in ACTION_VERBS):
        if "To-Do" not in buckets:
            buckets.append("To-Do")

    # "todo"/"tasks" header + numbered/bulleted list → To-Do
    if re.search(r'\b(todo|task|tasks|to-do)\b', lc) or (
        re.search(r'\n\s*\d+[.)]\s|\n\s*[-*•]\s', lc)
        and re.search(r'\b(today|tomorrow|this week)\b', lc)
    ):
        if "To-Do" not in buckets:
            buckets.append("To-Do")

    # Events — requires a specific time OR a named day-of-week.
    # "today" alone does NOT make something an Event; it just anchors a To-Do to today.
    has_specific_time = bool(re.search(
        r"\b(\d{1,2}\s*(am|pm)|\d{1,2}:\d{2}|noon|midnight|tonight|morning|afternoon|evening)\b", lc
    ))
    has_day_of_week = bool(re.search(
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", lc
    ))
    if has_specific_time or has_day_of_week:
        if "Events" not in buckets:
            buckets.append("Events")
        if "To-Do" not in buckets:
            buckets.append("To-Do")

    # Ideas
    if re.search(r"\b(idea|concept|what if|imagine|startup|feature|build|create)\b", lc):
        if "Ideas" not in buckets:
            buckets.append("Ideas")

    # Remember
    if re.search(r"\b(parked|password|code|pin|address|at|in|located)\b", lc):
        if "To-Do" not in buckets and "Remember" not in buckets:
            buckets.append("Remember")

    return list(dict.fromkeys(buckets)) or ["Random"]


def _extract_time_mention(content: str, ref: datetime) -> Tuple[Optional[str], Optional[str]]:
    lc = _normalize_time_words(content.lower())
    date_str: Optional[str] = None
    time_str: Optional[str] = None

    if "today" in lc:
        date_str = ref.strftime("%Y-%m-%d")
    elif "tomorrow" in lc:
        date_str = (ref + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        for i, d in enumerate(days):
            if d in lc:
                delta = (i - ref.weekday()) % 7 or 7
                date_str = (ref + timedelta(days=delta)).strftime("%Y-%m-%d")
                break

    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", lc)
    if m:
        h, mins, meridian = int(m.group(1)), int(m.group(2) or 0), m.group(3)
        if meridian == "pm" and h != 12:
            h += 12
        elif meridian == "am" and h == 12:
            h = 0
        time_str = f"{h:02d}:{mins:02d}"
    elif "noon" in lc:
        time_str = "12:00"
    elif "midnight" in lc:
        time_str = "00:00"
    else:
        # "at N" or "at N:MM" without am/pm — infer meridiem from context
        m_bare = re.search(r"\bat\s+(\d{1,2})(?::(\d{2}))?\b", lc)
        if m_bare:
            h    = int(m_bare.group(1))
            mins = int(m_bare.group(2) or 0)
            if _infer_meridiem(lc, h) and h != 12:
                h += 12
            time_str = f"{h:02d}:{mins:02d}"
        else:
            # Relative time: "in 30 minutes" / "in 2 hours"
            rel = re.search(r"\bin\s+(\d+)\s*(min|mins|minute|minutes)\b", lc)
            if rel:
                target = ref + timedelta(minutes=int(rel.group(1)))
                time_str = target.strftime("%H:%M")
                if not date_str:
                    date_str = target.strftime("%Y-%m-%d")
            else:
                rel = re.search(r"\bin\s+(\d+)\s*(hour|hours)\b", lc)
                if rel:
                    target = ref + timedelta(hours=int(rel.group(1)))
                    time_str = target.strftime("%H:%M")
                    if not date_str:
                        date_str = target.strftime("%Y-%m-%d")

    return date_str, time_str


# ─────────────────────────────────────────────────────────────────────────────
# Main processor
# ─────────────────────────────────────────────────────────────────────────────

class MessageProcessor:

    def __init__(self, cerebras_client: CerebrasClient, reminder_service=None):
        self.cerebras         = cerebras_client
        self.reminder_service = reminder_service
        self._list_service    = None  # injected lazily to avoid circular import

    @property
    def list_service(self):
        if self._list_service is None:
            from services.list_service import ListService
            self._list_service = ListService(self.cerebras)
        return self._list_service

    # ──────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────

    async def process(
        self,
        user_phone: str,
        content: str,
        message_type: str,
        db: AsyncSession,
        media_url: Optional[str] = None,
        skip_query: bool = False,
        group_id: Optional[int] = None,
        force_bucket: Optional[str] = None,
        no_llm_fallback: bool = False,
        skip_reminder: bool = False,
    ) -> Dict:
        user = await self._get_user(user_phone, db)
        if not user:
            raise ValueError("User not registered")

        ref = _ist_now()

        # ── Document capture: save immediately, extract text in background ──────
        # Returns instantly with caption + filename; a background asyncio task then
        # extracts the PDF/DOCX text and re-generates the embedding so the document
        # becomes keyword-searchable within a few seconds. main.py fires the background
        # task AFTER stamping file_name/is_document tags (to avoid a write race).
        if message_type == "document" and media_url:
            caption = (content or "").strip()
            return await self._save_document_minimal(
                user=user, caption=caption, media_url=media_url, db=db, ref=ref,
            )

        # ── Image capture: save immediately, run vision analysis in background ────
        # Returns instantly with caption + media_url; a background asyncio task then
        # runs Gemini Vision (OCR + recall keywords) and re-generates the embedding
        # so the image becomes keyword-searchable within a few seconds. The task is
        # fired by main.py immediately after this return (no write race since main.py
        # does no DB updates for images — just echoes media_url in the response).
        if message_type == "image" and media_url and not force_bucket:
            caption_text = (content or "").strip()
            result = await self._save_image_minimal(
                user=user, caption=caption_text, media_url=media_url, db=db, ref=ref,
            )
            return result

        # ── Media override: link captures → always Remember ──────────────────
        # Share-extension URL captures (message_type="text" + http media_url) skip
        # AI classification entirely; the user can move the bucket manually afterward.
        if not force_bucket and (
            message_type == "image"
            or (message_type == "text" and media_url)
        ):
            force_bucket = "Remember"
            print(f"[processor] media override → Remember (type={message_type}, has_url={bool(media_url)})")

        # ── FAST PATH: Regex list detection — skip when force_bucket is set ──
        # When force_bucket is set (e.g. group @mention → always To-Do) we skip
        # list detection to honour the caller's intent directly.
        if not force_bucket:
            regex_list = self.list_service._regex_detect(content)
            if regex_list and regex_list.get("intent") == "create_or_add" and regex_list.get("items"):
                if not _LIST_DATE_RE.search(content):
                    list_bucket = "Remember"
                    from services.classifier_service import classifier_service, CONF_THRESHOLD
                    if classifier_service.is_ready:
                        _b, _conf = classifier_service.classify(content)
                        if _conf >= CONF_THRESHOLD and _b in ("To-Do", "Remember", "Track"):
                            list_bucket = _b
                    print(f"[processor] regex_list hit: {regex_list['list_name']!r} ({len(regex_list['items'])} items), bucket={list_bucket}")
                    return await self._handle_list_save_direct(
                        user, regex_list["list_name"], regex_list["list_type"], regex_list["items"], db,
                        group_id=group_id, bucket=list_bucket,
                    )

        # ── Single LLM call: multi-action intent parse ────────────
        from services.intent_service import get_intent_service
        intent_svc = get_intent_service(self.cerebras)
        parsed     = await intent_svc.parse(
            content, user.name, user.timezone or "Asia/Kolkata",
            check_query=not skip_query,
            force_bucket=force_bucket,
            no_llm_fallback=no_llm_fallback,
        )

        actions = parsed.get("actions", {})
        print(f"[processor] actions={[k for k,v in actions.items() if v]} for: {content[:60]}")

        # ── Named list — highest priority, self-contained ─────────
        if actions.get("save_as_list") and parsed.get("list"):
            lst = parsed["list"]
            return await self._handle_list_save_direct(
                user, lst["list_name"], lst["list_type"], lst["items"], db,
                group_id=group_id,
                due_date=lst.get("due_date"),
                bucket=lst.get("bucket", "Remember"),
            )

        # ── Query — user wants to retrieve something ──────────────
        # skip_query=True when called from the iOS capture endpoint (dump tab always saves)
        # Also reclassify as todo if the LLM marked is_query but there's no "?" — imperative sentence
        if actions.get("is_query") and not skip_query and "?" in content:
            return {"_is_query": True, "query_data": parsed.get("query", {})}
        if actions.get("is_query") and not skip_query:
            # Imperative without "?" — treat as a todo for today
            actions["is_query"] = False
            actions["save_as_todo"] = True
            if not parsed.get("tasks"):
                parsed["tasks"] = [{"task": content, "due_date": _today_str(), "time": None, "priority": "normal"}]

        # ── Tasks (todo + optional event + optional reminder) ─────
        if actions.get("save_as_todo") and parsed.get("tasks"):
            tasks = parsed["tasks"]
            if len(tasks) > 1:
                return await self._process_tasks(
                    user=user, tasks=tasks, parsed=parsed,
                    original_content=content, db=db,
                    skip_reminder=skip_reminder,
                )
            elif len(tasks) == 1:
                return await self._process_single_task(
                    user=user, task=tasks[0], parsed=parsed,
                    content=content, message_type=message_type,
                    media_url=media_url, db=db, ref=ref,
                    skip_reminder=skip_reminder,
                )

        # ── Track ─────────────────────────────────────────────────
        # parsed["track"] may be absent when force_bucket="Track" (e.g. expense
        # chip) — _build_from_bucket sets save_as_track but doesn't populate
        # the track dict. Pass an empty dict; _process_track handles it fine
        # (falls back to content[:80] as summary).
        if actions.get("save_as_track"):
            return await self._process_track(
                user=user, track=parsed.get("track") or {}, content=content,
                message_type=message_type, media_url=media_url, db=db, ref=ref
            )

        # ── Note ──────────────────────────────────────────────────
        if actions.get("save_as_note"):
            note_data = parsed.get("note") or {"content": content, "keywords": []}
            return await self._save_single(
                user=user, content=content, bucket="Remember",
                keywords=note_data.get("keywords", []),
                message_type=message_type, media_url=media_url, db=db, ref=ref,
            )

        # ── Idea ──────────────────────────────────────────────────
        if actions.get("save_as_idea"):
            idea_data = parsed.get("idea") or {"content": content, "keywords": []}
            return await self._save_single(
                user=user, content=content, bucket="Ideas",
                keywords=idea_data.get("keywords", []),
                message_type=message_type, media_url=media_url, db=db, ref=ref
            )

        # ── Pure Event (no To-Do, no explicit reminder) ───────────
        # Fires for both LLM-parsed events (save_as_event=True, save_as_todo=False)
        # and the LLM-failure fallback (_build_from_bucket("Events")).
        if actions.get("save_as_event") and not actions.get("save_as_todo"):
            evt = parsed.get("event") or {}
            return await self._save_single_event(
                user=user, content=content,
                due_date=evt.get("due_date"), event_time=evt.get("time"),
                message_type=message_type, media_url=media_url, db=db, ref=ref,
                skip_reminder=skip_reminder,
            )

        # ── Fallback ──────────────────────────────────────────────
        # When intent LLM 429'd (all actions False), try pure-regex list
        # detection before saving as a generic note/random entry.
        if not any(actions.values()):
            list_result = self.list_service._regex_detect(content)
            if list_result:
                if list_result["intent"] == "create_or_add" and list_result.get("items"):
                    from services.intent_service import _infer_bucket_from_rules
                    fallback_bucket = _infer_bucket_from_rules(content)
                    return await self._handle_list_save_direct(
                        user,
                        list_result["list_name"],
                        list_result["list_type"],
                        list_result["items"],
                        db,
                        group_id=group_id,
                        bucket=fallback_bucket,
                    )
                elif list_result["intent"] == "show":
                    return {
                        "_is_query": True,
                        "query_data": {
                            "query_text": list_result["list_name"],
                            "date_hint":  None,
                            "list_name":  list_result["list_name"],
                        },
                    }

        return await self._process_fallback(
            user=user, content=content, message_type=message_type,
            media_url=media_url, db=db, ref=ref
        )


    # ──────────────────────────────────────────────────────────────
    # Handlers — consume multi-action parsed output
    # ──────────────────────────────────────────────────────────────

    async def _handle_list_save_direct(
        self, user, list_name: str, list_type: str, items: List[str], db: AsyncSession,
        group_id: Optional[int] = None,
        due_date: Optional[str] = None,
        bucket: str = "Remember",
    ) -> Dict:
        # Shopping/bag/reading/watching lists are recall collections — users look them
        # up at the store or when packing, they don't "check them off" as tasks.
        # Override any classifier-assigned To-Do bucket for these types.
        _RECALL_LIST_TYPES = {"shopping", "bag", "reading", "watching"}
        if list_type in _RECALL_LIST_TYPES and bucket == "To-Do":
            bucket = "Remember"

        # To-Do lists with no explicit date default to today so they appear
        # under TODAY in the iOS TodoView instead of SOMEDAY.
        if bucket == "To-Do" and not due_date:
            due_date = _today_str()

        msg, added, was_created = await self.list_service.create_or_add(
            user.id, list_name, list_type, items, db,
            group_id=group_id, due_date=due_date, bucket=bucket,
        )
        tags  = msg.tags if isinstance(msg.tags, dict) else {}
        total = len(tags.get("subtasks", []))
        stored_due = tags.get("due_date") or due_date
        actual_bucket = tags.get("primary_bucket", bucket)
        return {
            "message_id":  msg.id,
            "category":    actual_bucket,
            "all_buckets": [actual_bucket],
            "is_list":     True,
            "list_type":   list_type,
            "list_name":   list_name,
            "items_added": added,
            "total_items": total,
            "was_created": was_created,
            "tags":        [],
            "essence":     f"{list_name} — {added} item(s) added ({total} total)",
            "connections": [],
            "due_date":    stored_due,
            "events":      [],
            "priority":    "normal",
        }

    async def _process_tasks(
        self, user, tasks: List[Dict], parsed: Dict,
        original_content: str, db: AsyncSession,
        skip_reminder: bool = False,
    ) -> Dict:
        """
        Save multiple tasks. Each task is its own Message row.
        Each task with a time gets its own reminder.
        """
        category = await self._get_or_create_category(
            user_id=user.id, name="To-Do",
            auto_description=INTENT_BUCKETS["To-Do"], db=db,
        )

        saved_ids   = []
        saved_items = []
        ref         = _ist_now()
        people      = parsed.get("people", [])

        for task_data in tasks:
            task     = str(task_data.get("task", "")).strip()
            due_date = task_data.get("due_date") or _today_str()
            evt_time = task_data.get("time")
            priority = task_data.get("priority", "normal")

            if not task:
                continue
            if not _DATE_RE.match(str(due_date)):
                due_date = _today_str()

            buckets = ["To-Do"]
            if evt_time:
                buckets.append("Events")

            _task_will_remind = bool(evt_time and not skip_reminder)
            msg = Message(
                user_id=user.id,
                category_id=category.id,
                content=task,
                message_type=MessageType("text"),
                summary=task[:100],
                tags={
                    "keywords":       [task],
                    "entities":       {"people": people},
                    "actionables":    [task],
                    "sentiment":      "neutral",
                    "priority":       priority,
                    "time_reference": "today" if due_date == _today_str() else "future",
                    "event_time":     evt_time,
                    "all_buckets":    buckets,
                    "primary_bucket": "To-Do",
                    "due_date":       due_date,
                    "events": (
                        [{"date": due_date, "time": evt_time, "label": task[:40]}]
                        if evt_time else []
                    ),
                    "split_from": original_content[:200],
                    **({"is_reminder": True} if _task_will_remind else {}),
                },
                created_at=ref,
            )
            db.add(msg)
            await db.flush()

            saved_ids.append(msg.id)
            saved_items.append({
                "id": msg.id, "task": task,
                "evt_time": evt_time, "due_date": due_date,
                "priority": priority,
            })

        await db.commit()

        # Embeddings + per-task reminders
        reminder_count = 0
        for saved in saved_items:
            self._save_embedding(saved["id"], saved["task"], {}, db)
            # Each task with a time gets its OWN reminder (skip for group @mention captures)
            if self.reminder_service and saved["evt_time"] and not skip_reminder:
                try:
                    reminder = await self.reminder_service.create(
                        user=user,
                        content=saved["task"],
                        analysis={
                            "event_time": saved["evt_time"],
                            "due_date":   saved["due_date"],
                            "priority":   saved["priority"],
                            "actionables": [saved["task"]],
                            "essence":    saved["task"],
                        },
                        message_id=saved["id"],
                        db=db,
                    )
                    if reminder:
                        reminder_count += 1
                        from sqlalchemy import text as _text
                        await db.execute(
                            _text("UPDATE messages SET tags = tags || CAST(:extra AS jsonb) WHERE id = :mid")
                            .bindparams(extra=json.dumps({"remind_at": reminder.remind_at.isoformat()}), mid=saved["id"])
                        )
                        await db.commit()
                    else:
                        reminder_count += 1
                except Exception as e:
                    print(f"⚠ Reminder failed for '{saved['task']}': {e}")

        reminder_note = f" · {reminder_count} reminder(s) set" if reminder_count else ""

        return {
            "message_id":  saved_ids[0] if saved_ids else None,
            "split_count": len(saved_ids),
            "message_ids": saved_ids,
            "category":    "To-Do",
            "all_buckets": ["To-Do"],
            "tags":        [],
            "essence":     f"Saved {len(saved_ids)} tasks{reminder_note}: "
                          + ", ".join(s["task"] for s in saved_items[:3]),
            "connections": [],
            "due_date":    saved_items[0]["due_date"] if saved_items else None,
            "events":      [],
            "priority":    parsed.get("priority", "normal"),
        }

    async def _process_single_task(
        self, user, task: Dict, parsed: Dict, content: str,
        message_type: str, media_url: Optional[str], db: AsyncSession, ref: datetime,
        skip_reminder: bool = False,
    ) -> Dict:
        """Save a single task, optionally with reminder and event."""
        due_date  = task.get("due_date") or _today_str()
        evt_time  = task.get("time")
        priority  = task.get("priority", "normal")
        people    = parsed.get("people", [])
        actions   = parsed.get("actions", {})

        if not _DATE_RE.match(str(due_date)):
            due_date = _today_str()

        # When the classifier identified this as an event (save_as_event=True),
        # store it under the Events bucket while keeping To-Do in all_buckets so
        # it still appears in task lists.
        primary_bucket = "Events" if actions.get("save_as_event") else "To-Do"
        buckets = ["To-Do"]
        if actions.get("save_as_event") or evt_time:
            buckets.append("Events")

        category = await self._get_or_create_category(
            user_id=user.id, name=primary_bucket,
            auto_description=INTENT_BUCKETS.get(primary_bucket, INTENT_BUCKETS["To-Do"]), db=db,
        )

        will_remind = bool(actions.get("set_reminder") and evt_time and not skip_reminder)
        tags = {
            "keywords":       [content],
            "entities":       {"people": people},
            "actionables":    [content],
            "sentiment":      "neutral",
            "priority":       priority,
            "time_reference": "today" if due_date == _today_str() else "future",
            "event_time":     evt_time,
            "all_buckets":    buckets,
            "primary_bucket": primary_bucket,
            "due_date":       due_date,
            "events": (
                [{"date": due_date, "time": evt_time, "label": content[:40]}]
                if evt_time else []
            ),
            **({"is_reminder": True} if will_remind else {}),
        }

        # Always store the user's original text — never the LLM-rewritten task description.
        # due_date/priority/evt_time come from the parsed task dict; content stays verbatim.
        msg = Message(
            user_id=user.id,
            category_id=category.id,
            content=content,
            message_type=MessageType(message_type),
            media_url=media_url,
            summary=content[:100],
            tags=tags,
            created_at=ref,
        )
        db.add(msg)
        await db.commit()
        await db.refresh(msg)

        # Debug: confirm what was stored in DB tags
        stored_tags = msg.tags if isinstance(msg.tags, dict) else {}
        print(f"[task_save] id={msg.id} primary={primary_bucket} due={stored_tags.get('due_date')!r} "
              f"event_time={stored_tags.get('event_time')!r} task='{content[:50]}'")

        result = {
            "message_id":  msg.id,
            "category":    primary_bucket,
            "all_buckets": buckets,
            "tags":        [],
            "essence":     content,
            "connections": [],
            "due_date":    due_date,
            "event_time":  evt_time,
            "events":      tags["events"],
            "priority":    priority,
        }

        # Reminder — only if set_reminder=true AND time is present AND not delegated to caller
        if self.reminder_service and actions.get("set_reminder") and evt_time and not skip_reminder:
            try:
                rem_data = parsed.get("reminder") or {}
                reminder = await self.reminder_service.create(
                    user=user,
                    content=content,
                    analysis={
                        "event_time":  evt_time,
                        "due_date":    due_date,
                        "priority":    priority,
                        "actionables": [content],
                        "essence":     content,
                    },
                    message_id=msg.id,
                    db=db,
                )
                if reminder:
                    result["reminder_id"] = reminder.id
                    result["remind_at"]   = reminder.remind_at.isoformat()
                    from sqlalchemy import text as _text
                    await db.execute(
                        _text("UPDATE messages SET tags = tags || CAST(:extra AS jsonb) WHERE id = :mid")
                        .bindparams(extra=json.dumps({"remind_at": reminder.remind_at.isoformat()}), mid=msg.id)
                    )
                    await db.commit()
            except Exception as e:
                print(f"⚠ Reminder failed: {e}")

        self._save_embedding(msg.id, content, {}, db)
        return result

    async def _process_track(
        self, user, track: Dict, content: str, message_type: str,
        media_url: Optional[str], db: AsyncSession, ref: datetime
    ) -> Dict:
        logs    = track.get("logs", [])
        summary = ", ".join(
            f"{l.get('metric')} {l.get('value')}{l.get('unit','')}" for l in logs
        ) or content[:80]

        category = await self._get_or_create_category(
            user_id=user.id, name="Track",
            auto_description=INTENT_BUCKETS["Track"], db=db,
        )

        tags = {
            "all_buckets":    ["Track"],
            "primary_bucket": "Track",
            "logs":           logs,
            "priority":       "normal",
            "due_date":       ref.strftime("%Y-%m-%d"),
        }

        msg = Message(
            user_id=user.id, category_id=category.id,
            content=content, message_type=MessageType(message_type),
            media_url=media_url, summary=summary, tags=tags, created_at=ref,
        )
        db.add(msg)
        await db.commit()
        await db.refresh(msg)
        self._save_embedding(msg.id, content, {}, db)

        return {
            "message_id": msg.id, "category": "Track",
            "all_buckets": ["Track"], "tags": [],
            "essence": summary, "connections": [],
            "due_date": None, "events": [], "priority": "normal",
        }

    async def _save_single(
        self, user, content: str, bucket: str, keywords: List[str],
        message_type: str, media_url: Optional[str], db: AsyncSession, ref: datetime,
        embed_text: Optional[str] = None,
        summary: Optional[str] = None,
    ) -> Dict:
        category = await self._get_or_create_category(
            user_id=user.id, name=bucket,
            auto_description=INTENT_BUCKETS.get(bucket, ""), db=db,
        )

        tags = {
            "keywords":       keywords,
            "entities":       {},
            "all_buckets":    [bucket],
            "primary_bucket": bucket,
            "priority":       "normal",
            "due_date":       None,
        }
        _sk = _implicit_search_keywords(message_type, media_url)
        if _sk:
            tags["search_keywords"] = _sk

        msg = Message(
            user_id=user.id, category_id=category.id,
            content=content, message_type=MessageType(message_type),
            media_url=media_url, summary=summary if summary is not None else content[:100],
            tags=tags, created_at=ref,
        )
        db.add(msg)
        await db.commit()
        await db.refresh(msg)
        # For documents, embed_text is the focused lead; otherwise embed the content.
        self._save_embedding(msg.id, embed_text or content, {"keywords": keywords}, db)

        return {
            "message_id": msg.id, "category": bucket,
            "all_buckets": [bucket], "tags": keywords,
            "essence": content[:100], "connections": [],
            "due_date": None, "events": [], "priority": "normal",
        }

    async def _save_single_event(
        self, user, content: str, due_date: Optional[str],
        event_time: Optional[str], message_type: str,
        media_url: Optional[str], db: AsyncSession, ref: datetime,
        skip_reminder: bool = False,
    ) -> Dict:
        if not due_date:
            due_date = ref.strftime("%Y-%m-%d")

        category = await self._get_or_create_category(
            user_id=user.id, name="Events",
            auto_description=INTENT_BUCKETS["Events"], db=db,
        )

        events = [{"date": due_date, "time": event_time, "label": content[:40]}]
        tags = {
            "all_buckets":    ["Events"],
            "primary_bucket": "Events",
            "priority":       "normal",
            "due_date":       due_date,
            "event_time":     event_time,
            "events":         events,
        }

        # Schedule a silent day-before push (no Reminder row — won't appear in Reminders section).
        # Store the notification date in tags; the scheduler fires APNs when date matches today.
        if not skip_reminder:
            from datetime import date as _date
            try:
                event_date = _date.fromisoformat(due_date)
                notify_date = event_date - timedelta(days=1)
                from zoneinfo import ZoneInfo
                today_in_tz = datetime.now(ZoneInfo(user.timezone or "Asia/Kolkata")).date()
                if notify_date >= today_in_tz:
                    tags["auto_notify_date"] = notify_date.isoformat()
            except Exception:
                pass

        msg = Message(
            user_id=user.id, category_id=category.id,
            content=content, message_type=MessageType(message_type),
            media_url=media_url, summary=content[:100], tags=tags, created_at=ref,
        )
        db.add(msg)
        await db.commit()
        await db.refresh(msg)
        self._save_embedding(msg.id, content, {}, db)

        return {
            "message_id":  msg.id,
            "category":    "Events",
            "all_buckets": ["Events"],
            "tags":        [],
            "essence":     content[:100],
            "connections": [],
            "due_date":    due_date,
            "events":      events,
            "priority":    "normal",
        }

    async def _process_fallback(
        self, user, content: str, message_type: str,
        media_url: Optional[str], db: AsyncSession, ref: datetime
    ) -> Dict:
        """Old _full_analysis path as safety net."""
        fast_buckets = _sniff_buckets_fast(content)
        fast_date, fast_time = _extract_time_mention(content, ref)

        analysis = await self._full_analysis(
            content=content, user=user, db=db,
            fast_buckets=fast_buckets, fast_date=fast_date,
            fast_time=fast_time, ref=ref,
        )

        buckets = [b for b in analysis.get("buckets", fast_buckets) if b in BUCKET_NAMES]
        if not buckets:
            buckets = ["Random"]

        primary_bucket = buckets[0]
        category = await self._get_or_create_category(
            user_id=user.id, name=primary_bucket,
            auto_description=INTENT_BUCKETS.get(primary_bucket, ""), db=db,
        )

        due_date = analysis.get("due_date")
        if not due_date and "To-Do" in buckets:
            due_date = _today_str()
        if not due_date and analysis.get("event_time"):
            due_date = _today_str()

        _will_remind_fallback = bool(
            ("To-Do" in buckets or "Events" in buckets)
            and analysis.get("event_time") and due_date
        )
        tags = {
            "keywords":       analysis.get("keywords", []),
            "entities":       analysis.get("entities", {}),
            "concepts":       analysis.get("concepts", []),
            "actionables":    analysis.get("actionables", []),
            "sentiment":      analysis.get("sentiment", "neutral"),
            "priority":       analysis.get("priority", "normal"),
            "time_reference": analysis.get("time_reference", "none"),
            "event_time":     analysis.get("event_time"),
            "all_buckets":    buckets,
            "primary_bucket": primary_bucket,
            "due_date":       due_date,
            "events":         analysis.get("events", []),
            **({"is_reminder": True} if _will_remind_fallback else {}),
        }

        msg = Message(
            user_id=user.id, category_id=category.id,
            content=content, message_type=MessageType(message_type),
            media_url=media_url,
            summary=analysis.get("essence", content[:100]),
            tags=tags, created_at=ref,
        )
        db.add(msg)
        await db.commit()
        await db.refresh(msg)

        result = {
            "message_id":  msg.id, "category": primary_bucket,
            "all_buckets": buckets, "tags": analysis.get("keywords", []),
            "essence":     analysis.get("essence", ""),
            "connections": analysis.get("related_concepts", []),
            "due_date":    due_date, "events": analysis.get("events", []),
            "priority":    analysis.get("priority", "normal"),
        }

        if ("To-Do" in buckets or "Events" in buckets) and self.reminder_service:
            if analysis.get("event_time") and due_date:
                try:
                    reminder = await self.reminder_service.create(
                        user=user, content=content, analysis=analysis,
                        message_id=msg.id, db=db,
                    )
                    if reminder:
                        result["reminder_id"] = reminder.id
                        result["remind_at"]   = reminder.remind_at.isoformat()
                        from sqlalchemy import text as _text
                        await db.execute(
                            _text("UPDATE messages SET tags = tags || CAST(:extra AS jsonb) WHERE id = :mid")
                            .bindparams(extra=json.dumps({"remind_at": reminder.remind_at.isoformat()}), mid=msg.id)
                        )
                        await db.commit()
                except Exception as e:
                    print(f"⚠ Reminder failed: {e}")

        self._save_embedding(msg.id, content, analysis, db)
        return result


        # ──────────────────────────────────────────────────────────────
    # List save handler
    # ──────────────────────────────────────────────────────────────

    async def _handle_list_save(
        self, user: User, intent: Dict, original_content: str, db: AsyncSession
    ) -> Dict:
        list_type = intent["list_type"]
        list_name = intent["list_name"]
        items     = intent["items"]

        msg, added, was_created = await self.list_service.create_or_add(
            user.id, list_name, list_type, items, db
        )

        tags  = msg.tags if isinstance(msg.tags, dict) else {}
        total = len(tags.get("subtasks", []))

        return {
            "message_id":  msg.id,
            "category":    "List",
            "all_buckets": ["List"],
            "list_type":   list_type,
            "list_name":   list_name,
            "items_added": added,
            "total_items": total,
            "tags":        [],
            "essence":     f"{list_name} — {added} item(s) added ({total} total)",
            "connections": [],
            "due_date":    None,
            "events":      [],
            "priority":    "normal",
        }

    # ──────────────────────────────────────────────────────────────
    # LLM analysis
    # ──────────────────────────────────────────────────────────────

    async def _full_analysis(
        self, content: str, user: User, db: AsyncSession,
        fast_buckets: List[str], fast_date: Optional[str],
        fast_time: Optional[str], ref: datetime,
    ) -> Dict:
        recent_topics = await self._get_recent_topics(user.id, db)
        ref_str  = ref.strftime("%Y-%m-%d (%A, %d %B %Y)")
        tomorrow = (ref + timedelta(days=1)).strftime("%Y-%m-%d")

        bucket_defs = "\n".join(
            f'  "{name}": {desc}' for name, desc in INTENT_BUCKETS.items()
        )

        prompt = f"""You are analyzing a personal note saved by {user.name}.
TODAY: {ref_str}
TOMORROW: {tomorrow}
RECENT TOPICS: {", ".join(recent_topics[:10])}

NOTE: "{content}"

BUCKET DEFINITIONS (choose 1-3):
{bucket_defs}

MULTI-BUCKET RULE:
  "call Kailash at 12pm tomorrow" → ["To-Do", "Events"]
  "buy milk" → ["To-Do"]
  "parked at C3" → ["Remember"]
  "weight 74kg" → ["Track"]
  "dentist Friday 3pm" → ["Events"]
  "startup idea" → ["Ideas"]
  "heyy" → ["Random"]

PRIORITY:
  urgent/asap/critical/important → "high"
  emergency/immediately → "urgent"
  normal → "normal"

FAST HINTS: fast_buckets={fast_buckets}, fast_date={fast_date}, fast_time={fast_time}
DATE: today={ref.strftime("%Y-%m-%d")}, tomorrow={tomorrow}

Return ONLY this JSON:
{{
  "buckets": ["Primary"],
  "essence": "one clear sentence",
  "keywords": ["3-6 keywords"],
  "entities": {{
    "people": [], "places": [], "organizations": [],
    "products": [], "times": [], "phone_numbers": [], "emails": [], "urls": []
  }},
  "concepts":       [],
  "actionables":    [],
  "sentiment":      "neutral",
  "priority":       "normal",
  "time_reference": "today | tomorrow | this_week | future | none",
  "event_time":     "HH:MM or null",
  "due_date":       "YYYY-MM-DD or null",
  "events": [{{"date": "YYYY-MM-DD", "time": "HH:MM or null", "label": "short title"}}],
  "todo_items": [{{"task": "clean task", "due_date": "YYYY-MM-DD", "time": "HH:MM or null"}}],
  "related_concepts": []
}}"""

        try:
            response = await self.cerebras.chat(prompt, max_tokens=1500)
        except Exception as e:
            print(f"[processor] _full_analysis LLM failed: {e}")
            response = {}

        response.setdefault("buckets", fast_buckets)
        response.setdefault("essence", content[:100])
        response.setdefault("keywords", [])
        response.setdefault("entities", {})
        response.setdefault("concepts", [])
        response.setdefault("actionables", [])
        response.setdefault("sentiment", "neutral")
        response.setdefault("priority", "normal")
        response.setdefault("time_reference", "none")
        response.setdefault("event_time", None)
        response.setdefault("due_date", None)
        response.setdefault("events", [])
        response.setdefault("todo_items", [])
        response.setdefault("related_concepts", [])

        # When LLM didn't extract time/date, promote the regex-derived values.
        # This ensures reminders fire correctly even when the LLM is unavailable.
        if not response["event_time"] and fast_time:
            response["event_time"] = fast_time
        if not response["due_date"] and fast_date:
            response["due_date"] = fast_date

        ent = response["entities"]
        if not isinstance(ent, dict):
            response["entities"] = {}
        for k in ("people","places","organizations","products","times","phone_numbers","emails","urls"):
            response["entities"].setdefault(k, [])

        dd = response.get("due_date")
        if dd and not _DATE_RE.match(str(dd)):
            response["due_date"] = None

        valid_events = []
        for e in response.get("events", []):
            if isinstance(e, dict) and _DATE_RE.match(str(e.get("date", ""))):
                valid_events.append({
                    "date":  e["date"],
                    "time":  e.get("time"),
                    "label": str(e.get("label", "Event"))[:60],
                })
        response["events"] = valid_events

        return response

    # ──────────────────────────────────────────────────────────────
    # Todo batch
    # ──────────────────────────────────────────────────────────────

    async def _process_todo_batch(
        self, user: User, items: List[Dict], original_content: str,
        analysis: Dict, all_buckets: List[str], db: AsyncSession,
    ) -> Dict:
        category = await self._get_or_create_category(
            user_id=user.id, name="To-Do",
            auto_description=INTENT_BUCKETS["To-Do"], db=db,
        )

        saved_ids   = []
        saved_items = []
        ref         = _ist_now()

        for item in items:
            task     = str(item.get("task", "")).strip()
            # Use item due_date, then analysis-level due_date (e.g. "tomorrow"), then today
            due_date = (
                item.get("due_date")
                or analysis.get("due_date")
                or _today_str()
            )
            evt_time = item.get("time")
            priority = _detect_priority(task) or analysis.get("priority", "normal")

            if not task:
                continue
            if not _DATE_RE.match(str(due_date)):
                due_date = _today_str()

            item_buckets = ["To-Do"]
            if evt_time:
                item_buckets.append("Events")

            message = Message(
                user_id=user.id,
                category_id=category.id,
                content=task,
                message_type=MessageType("text"),
                summary=task[:100],
                tags={
                    "keywords":       analysis.get("keywords", []),
                    "entities":       analysis.get("entities", {}),
                    "concepts":       analysis.get("concepts", []),
                    "actionables":    [task],
                    "sentiment":      "neutral",
                    "priority":       priority,
                    "time_reference": "none",
                    "event_time":     evt_time,
                    "all_buckets":    item_buckets,
                    "primary_bucket": "To-Do",
                    "due_date":       due_date,
                    "events": (
                        [{"date": due_date, "time": evt_time, "label": task[:40]}]
                        if "Events" in item_buckets else []
                    ),
                    "split_from": original_content[:200],
                    **({"is_reminder": True} if evt_time else {}),
                },
                created_at=ref,
            )
            db.add(message)
            await db.flush()

            saved_ids.append(message.id)
            saved_items.append({
                "id": message.id, "task": task,
                "evt_time": evt_time, "due_date": due_date,
                "priority": priority,
            })

        await db.commit()

        for saved in saved_items:
            self._save_embedding(saved["id"], saved["task"], analysis, db)

            if self.reminder_service and saved["evt_time"]:
                item_analysis = {
                    **analysis,
                    "event_time": saved["evt_time"],
                    "due_date":   saved["due_date"],
                    "priority":   saved["priority"],
                }
                try:
                    reminder = await self.reminder_service.create(
                        user=user, content=saved["task"],
                        analysis=item_analysis, message_id=saved["id"], db=db,
                    )
                    if reminder:
                        from sqlalchemy import text as _text
                        await db.execute(
                            _text("UPDATE messages SET tags = tags || CAST(:extra AS jsonb) WHERE id = :mid")
                            .bindparams(extra=json.dumps({"remind_at": reminder.remind_at.isoformat()}), mid=saved["id"])
                        )
                        await db.commit()
                except Exception as e:
                    print(f"⚠ Reminder failed for '{saved['task']}': {e}")

        timed   = [s for s in saved_items if s["evt_time"]]
        untimed = [s for s in saved_items if not s["evt_time"]]
        reminder_note = f" · {len(timed)} reminder(s) set" if timed else ""

        return {
            "message_id":  saved_ids[0] if saved_ids else None,
            "split_count": len(saved_ids),
            "message_ids": saved_ids,
            "category":    "To-Do",
            "all_buckets": ["To-Do"],
            "tags":        analysis.get("keywords", []),
            "essence":     (
                f"Saved {len(saved_ids)} tasks{reminder_note}: "
                + ", ".join(i["task"] for i in saved_items[:3])
            ),
            "connections": [],
            "due_date":    items[0].get("due_date") if items else None,
            "events":      [],
            "priority":    analysis.get("priority", "normal"),
        }

    # ──────────────────────────────────────────────────────────────
    # DB helpers
    # ──────────────────────────────────────────────────────────────

    async def _get_or_create_category(
        self, user_id: int, name: str, auto_description: str, db: AsyncSession,
    ) -> Category:
        result = await db.execute(
            select(Category).where(
                Category.user_id == user_id, Category.name == name,
            )
        )
        category = result.scalar_one_or_none()
        if not category:
            category = Category(user_id=user_id, name=name, description=auto_description)
            db.add(category)
            await db.flush()
        return category

    async def _get_user(self, phone: str, db: AsyncSession) -> Optional[User]:
        result = await db.execute(select(User).where(User.phone_number == phone))
        return result.scalar_one_or_none()

    async def _load_stored_bytes(self, media_url: str, db: AsyncSession) -> Optional[bytes]:
        """Load raw bytes for a /api/images/{id} media_url straight from the DB."""
        m = re.search(r"/api/images/(\d+)", media_url or "")
        if not m:
            return None
        from database import StoredImage
        img = await db.get(StoredImage, int(m.group(1)))
        return img.data if img else None

    async def _get_recent_topics(self, user_id: int, db: AsyncSession) -> List[str]:
        result = await db.execute(
            select(Message.tags)
            .where(Message.user_id == user_id)
            .order_by(Message.created_at.desc())
            .limit(30)
        )
        topics: set = set()
        for (tags,) in result.all():
            if isinstance(tags, dict):
                for kw in tags.get("keywords", [])[:3]:
                    topics.add(kw)
        return list(topics)[:15]

    def _save_embedding(
        self, message_id: int, content: str, analysis: Dict,
        db: AsyncSession = None,   # kept for call-site compat; ignored (task creates own session)
    ) -> None:
        """Schedule embedding generation as a fire-and-forget background task.

        Synchronous so callers don't need await — returns instantly.
        The embedding is written to the DB a few seconds later by
        _do_save_embedding(), which creates its own short-lived session.
        This prevents the Gemini API call from blocking the HTTP response.
        """
        import asyncio as _aio
        _aio.create_task(self._do_save_embedding(message_id, content, analysis))

    async def _do_save_embedding(self, message_id: int, content: str, analysis: Dict):
        """Background: call Gemini, write embedding to DB, close session."""
        import asyncio as _aio
        from database import async_session_maker
        from services.embedding_service import embedding_service
        people      = analysis.get("entities", {}).get("people", [])
        keywords    = analysis.get("keywords", [])
        essence     = analysis.get("essence", "")
        buckets     = analysis.get("buckets", [])
        actionables = analysis.get("actionables", [])

        enriched = " ".join(filter(None, [
            content, essence,
            " ".join(keywords), " ".join(people),
            " ".join(actionables), " ".join(buckets),
        ])).strip()

        for attempt in range(2):
            try:
                embedding = await embedding_service.aembed(
                    enriched, task_type="RETRIEVAL_DOCUMENT"
                )
                async with async_session_maker() as db:
                    await db.execute(
                        update(Message).where(Message.id == message_id).values(embedding=embedding)
                    )
                    await db.commit()
                return
            except Exception as e:
                if attempt == 0:
                    print(f"⚠ Embedding attempt 1 failed, retrying in 1s: {e}")
                    await _aio.sleep(1)
                else:
                    print(f"⚠ Embedding failed (non-critical): {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Background-enrichment helpers (image + document)
    # ─────────────────────────────────────────────────────────────────────────

    async def _save_image_minimal(
        self,
        user: User,
        caption: str,
        media_url: str,
        db: AsyncSession,
        ref: datetime,
    ) -> Dict:
        """Save an image message immediately with just the caption (no vision I/O).
        The vision analysis background task enriches the record seconds later."""
        category = await self._get_or_create_category(
            user_id=user.id, name="Remember",
            auto_description=INTENT_BUCKETS["Remember"], db=db,
        )
        tags: Dict = {
            "all_buckets":    ["Remember"],
            "primary_bucket": "Remember",
            "priority":       "normal",
            "due_date":       None,
            "search_keywords": "image photo pic picture",
        }
        if caption:
            tags["caption"] = caption

        msg = Message(
            user_id=user.id,
            category_id=category.id,
            content=caption or "",
            message_type=MessageType("image"),
            media_url=media_url,
            summary=caption[:100] if caption else None,
            tags=tags,
            created_at=ref,
        )
        db.add(msg)
        await db.commit()
        await db.refresh(msg)

        # Seed embedding from caption so the message is searchable immediately;
        # background task replaces this with a richer vector once vision runs.
        if caption:
            self._save_embedding(msg.id, caption, {}, db)

        return {
            "message_id":  msg.id,
            "category":    "Remember",
            "all_buckets": ["Remember"],
            "tags":        [],
            "essence":     caption or "",
            "connections": [],
            "due_date":    None,
            "events":      [],
            "priority":    "normal",
            "media_url":   media_url,
        }

    async def _save_document_minimal(
        self,
        user: User,
        caption: str,
        media_url: str,
        db: AsyncSession,
        ref: datetime,
    ) -> Dict:
        """Save a document message immediately with just the caption (no text extraction).
        main.py stamps file_name/is_document tags after this returns, then fires the
        background extraction task."""
        category = await self._get_or_create_category(
            user_id=user.id, name="Remember",
            auto_description=INTENT_BUCKETS["Remember"], db=db,
        )
        tags: Dict = {
            "all_buckets":    ["Remember"],
            "primary_bucket": "Remember",
            "priority":       "normal",
            "due_date":       None,
            "search_keywords": "document file pdf",
        }

        msg = Message(
            user_id=user.id,
            category_id=category.id,
            content=caption or "",
            message_type=MessageType("document"),
            media_url=media_url,
            summary=caption[:100] if caption else None,
            tags=tags,
            created_at=ref,
        )
        db.add(msg)
        await db.commit()
        await db.refresh(msg)

        if caption:
            self._save_embedding(msg.id, caption, {}, db)

        return {
            "message_id":  msg.id,
            "category":    "Remember",
            "all_buckets": ["Remember"],
            "tags":        [],
            "essence":     caption or "",
            "connections": [],
            "due_date":    None,
            "events":      [],
            "priority":    "normal",
            "media_url":   media_url,
        }

    async def _enrich_image_background(
        self,
        message_id: int,
        user_id: int,
        media_url: str,
        caption: str,
        mime_type: str,
    ) -> None:
        """Background task: run Gemini Vision then update content, tags, and embedding."""
        import asyncio as _aio
        from database import async_session_maker
        from sqlalchemy.ext.asyncio import AsyncSession as _AS
        from services.vision_service import vision_service
        from services.redis_cache import cache_del, bootstrap_key

        try:
            async with async_session_maker() as db:
                raw = await self._load_stored_bytes(media_url, db)
                if not raw:
                    print(f"[enrich] image {message_id}: bytes not found, skipping")
                    return

                print(f"[enrich] image {message_id}: running vision ({len(raw)//1024} KB)")
                try:
                    analysis = await vision_service.analyze_image(raw, mime_type)
                except Exception as e:
                    print(f"[enrich] image {message_id} vision failed: {e}")
                    analysis = {
                        "document_type": "other", "title": caption or "Image",
                        "extracted_text": "", "recall_terms": "", "description": "",
                    }

                extracted    = (analysis.get("extracted_text") or "").strip()
                recall_terms = (analysis.get("recall_terms")   or "").strip()
                description  = (analysis.get("description")    or "").strip()
                title        = (analysis.get("title") or caption or "Image").strip()
                doc_type     = analysis.get("document_type", "other")
                doc_label    = doc_type.replace("_", " ")

                # Keyword-searchable content (ILIKE hits on any term)
                kw: list[str] = []
                if caption:
                    kw.append(caption)
                # Only include the document type label when it's meaningful (not the
                # fallback "other" that vision returns on failure — it pollutes search).
                if doc_type and doc_type != "other":
                    kw.append(doc_label)
                if extracted:
                    kw.append(extracted[:5000])
                if recall_terms:
                    kw.append(recall_terms)
                content = "\n".join(kw)

                # Focused embed text (sharp semantic vector, not diluted by OCR noise)
                em: list[str] = []
                if caption:
                    em.append(caption)
                if description:
                    em.append(description)
                if recall_terms:
                    em.append(recall_terms)
                embed_text = " ".join(em) or content[:500]

                msg = await db.get(Message, message_id)
                if not msg:
                    return

                msg.content = content or caption or ""
                new_tags = dict(msg.tags or {})
                new_tags["image_title"]   = title
                new_tags["description"]   = description
                new_tags["recall_terms"]  = recall_terms
                new_tags["document_type"] = doc_type
                msg.tags = new_tags
                await db.commit()

                self._save_embedding(message_id, embed_text, {}, db)

                # Bust personal bootstrap so brain view gets fresh content
                await cache_del(bootstrap_key(user_id, None))
                print(f"[enrich] image {message_id}: done (OCR {len(extracted)} chars, recall_terms {len(recall_terms)} chars)")

        except Exception as e:
            print(f"[enrich] image {message_id} error: {e}")

    async def _enrich_document_background(
        self,
        message_id: int,
        user_id: int,
        media_url: str,
        caption: str,
    ) -> None:
        """Background task: extract PDF/DOCX text then update content and embedding.
        Does NOT touch tags — main.py already stamped file_name/is_document/caption."""
        from database import async_session_maker
        from services.document_processor import extract_text_from_bytes
        from services.redis_cache import cache_del, bootstrap_key

        try:
            async with async_session_maker() as db:
                raw = await self._load_stored_bytes(media_url, db)
                if not raw:
                    print(f"[enrich] doc {message_id}: bytes not found, skipping")
                    return

                print(f"[enrich] doc {message_id}: extracting text ({len(raw)//1024} KB)")
                try:
                    extracted = (await extract_text_from_bytes(raw, hint=caption or media_url) or "").strip()
                except Exception as e:
                    print(f"[enrich] doc {message_id} extraction error: {e}")
                    return

                if not extracted or extracted.startswith("["):
                    print(f"[enrich] doc {message_id}: no extractable text (image-only PDF?)")
                    return

                # Distil extracted text to 50-100 high-signal keywords via Gemini.
                # Storing raw 10k chars makes ILIKE search imprecise and embeds a diffuse vector.
                # Keywords are compact, Hinglish-friendly, and directly reflect what a user would
                # type when searching for this document.
                keywords = await self._extract_document_keywords(extracted, caption or "")
                searchable = (f"{caption}\n{keywords}" if caption else keywords).strip()
                embed_text = searchable

                msg = await db.get(Message, message_id)
                if not msg:
                    return
                msg.content = searchable
                await db.commit()

                self._save_embedding(message_id, embed_text, {}, db)
                await cache_del(bootstrap_key(user_id, None))
                print(f"[enrich] doc {message_id}: done ({len(extracted)} chars → {len(keywords)} keyword chars)")

        except Exception as e:
            print(f"[enrich] doc {message_id} error: {e}")

    async def _extract_document_keywords(self, text: str, caption: str) -> str:
        """Use Gemini to distil long extracted document text to 50-100 searchable keywords.
        Returns a space/newline-separated keyword string suitable for ILIKE search and embedding."""
        snippet = text[:6000]  # give Gemini enough context without burning tokens
        prompt = (
            "You are a search-keyword extractor for a personal knowledge app used by Indian professionals.\n"
            "Given the document text below, output ONLY a flat list of 50-100 high-signal search keywords "
            "and short phrases (2-3 words max each), one per line. No bullet points, no numbering, no explanations.\n"
            "Rules:\n"
            "- Include proper nouns: people, places, companies, products, amounts, dates\n"
            "- Include domain terms: technical jargon, category labels, document type\n"
            "- Include Hinglish variants where natural (e.g. 'ghar', 'kharcha', 'bill')\n"
            "- No stop words, no full sentences\n"
            f"Caption: {caption}\n\n"
            f"Document text:\n{snippet}\n\n"
            "Keywords:"
        )
        try:
            client = CerebrasClient(provider="gemini", model="gemini-2.5-flash-lite")
            raw = await client.chat(prompt, max_tokens=400)
            lines = [ln.strip().lower() for ln in raw.splitlines() if ln.strip()]
            return "\n".join(lines[:100])
        except Exception as e:
            print(f"[keywords] extraction failed, using lead: {e}")
            return text[:800]

    async def _enrich_link_background(
        self,
        message_id: int,
        user_id: int,
        link_url: str,
        caption: str,
    ) -> None:
        """Background task: fetch link OG metadata, extract keywords via Gemini,
        update content + embedding so the capture becomes semantically searchable."""
        from database import async_session_maker
        from services.redis_cache import cache_del, bootstrap_key
        import re as _re

        try:
            import httpx
        except ImportError:
            print(f"[enrich] link {message_id}: httpx not available, skipping")
            return

        try:
            async with async_session_maker() as db:
                # Fetch the page with a standard browser UA to get OG tags
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
                    ),
                    "Accept-Language": "en-US,en;q=0.9",
                }
                try:
                    async with httpx.AsyncClient(follow_redirects=True, timeout=10,
                                                  verify=False) as client:
                        resp = await client.get(link_url, headers=headers)
                        html = resp.text
                except Exception as e:
                    print(f"[enrich] link {message_id}: fetch failed: {e}")
                    return

                # Extract OG title + description from HTML
                def _og(prop: str) -> str:
                    m = _re.search(
                        rf'<meta[^>]+property=["\']og:{prop}["\'][^>]+content=["\']([^"\']+)["\']',
                        html, _re.IGNORECASE
                    ) or _re.search(
                        rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:{prop}["\']',
                        html, _re.IGNORECASE
                    )
                    return m.group(1).strip() if m else ""

                og_title = _og("title") or _og("site_name")
                og_desc  = _og("description")
                page_text = f"{og_title}\n{og_desc}".strip()

                if not page_text:
                    print(f"[enrich] link {message_id}: no OG metadata found")
                    return

                print(f"[enrich] link {message_id}: got OG text ({len(page_text)} chars)")
                keywords = await self._extract_document_keywords(page_text, caption or "")
                searchable = (f"{caption}\n{keywords}" if caption else keywords).strip()

                msg = await db.get(Message, message_id)
                if not msg:
                    return
                msg.content = searchable
                await db.commit()

                self._save_embedding(message_id, searchable, {}, db)
                await cache_del(bootstrap_key(user_id, None))
                print(f"[enrich] link {message_id}: done ({len(keywords)} keyword chars)")

        except Exception as e:
            print(f"[enrich] link {message_id} error: {e}")