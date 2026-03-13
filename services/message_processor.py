"""
Intelligent Message Processor — v2
────────────────────────────────────────────────────────────────────────────
KEY UPGRADES OVER v1
  • Multi-bucket assignment  — a note can live in MULTIPLE buckets at once
    e.g. "call Kailash at 12pm tomorrow" → To-Do + Events
  • Richer entity extraction — phone, time, URL, product, emotion, priority
  • Computed due_date always present (fallback = today for To-Do)
  • Action-verb detection for smarter To-Do vs Remember separation
  • Embeddings at 3072 dims (full Gemini gemini-embedding-001)
  • All LLM calls are single-pass where possible (fewer API round-trips)
────────────────────────────────────────────────────────────────────────────
Save as: services/message_processor.py
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
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
        "Examples: parked at C3, nailcutter in 3rd drawer, wifi password 1234, "
        "mom's birthday June 3rd, meeting room 4B."
    ),
    "To-Do": (
        "A task, action item, or reminder the user must act on. "
        "Examples: buy milk, call dentist, submit report by Friday, "
        "renew passport, send invoice, call Kailash at noon tomorrow."
    ),
    "Ideas": (
        "A new idea, thought, concept, insight, or creative spark. "
        "Examples: startup idea, feature suggestion, book concept, "
        "blog post topic, interesting question to explore."
    ),
    "Track": (
        "Log or monitor something over time — health, habits, progress, mood. "
        "Examples: weight 74 kg, ran 5 km, mood anxious, slept 6 hours, steps 8000."
    ),
    "Events": (
        "A time-anchored event, appointment, meeting, or plan with a specific date/time. "
        "Examples: mom visiting 10th March, dentist appointment Friday 3 pm, "
        "flight to Delhi on 20th, call Kailash at 12 pm tomorrow."
    ),
    "Random": (
        "Casual, venting, conversational, or clearly unclear intent. "
        "Examples: heyy, lol okay, ugh today was rough, testing 123, hi."
    ),
}

BUCKET_NAMES = list(INTENT_BUCKETS.keys())

# Signals that strongly imply To-Do regardless of other buckets
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
    "need to", "have to", "must", "should",
}

TRACK_SIGNALS = {
    "kg", "km", "mile", "steps", "calories", "kcal",
    "hours", "mins", "mood", "sleep", "slept", "ran",
    "walked", "weight", "bp", "sugar", "water", "drank",
}


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers
# ─────────────────────────────────────────────────────────────────────────────

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _today_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


def _tomorrow_str() -> str:
    return (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")


REMINDER_KEYWORDS = {"remind", "reminder", "don't forget", "alert", "notify", "ping"}

def _sniff_buckets_fast(content: str) -> List[str]:
    """
    Zero-LLM fast pre-classification.
    Returns a list of LIKELY buckets (may be wrong — LLM will refine).
    Used to warm up the prompt with hints.
    """
    lc = content.lower()

     # Reminder override — if explicit remind keyword, force To-Do first
    if any(kw in lc for kw in REMINDER_KEYWORDS):
        buckets = ["To-Do"]
        if re.search(r"\b(today|tomorrow|tonight|morning|afternoon|evening|noon|\d{1,2}(am|pm)|\d{1,2}:\d{2}|after\s+\d+\s*(min|hour))\b", lc):
            buckets.append("Events")
        return buckets


    buckets: List[str] = []

    words = set(re.findall(r"\b\w+\b", lc))

    # Random: very short or pure greeting
    if len(lc.strip()) < 6 or lc.strip() in {"hi", "hey", "heyy", "ok", "okay", "lol", "test"}:
        return ["Random"]

    # Track: numeric + health unit
    if any(sig in lc for sig in TRACK_SIGNALS) and re.search(r"\d", lc):
        buckets.append("Track")

    # To-Do: action verb at start
    if any(lc.startswith(v) or f" {v} " in lc for v in ACTION_VERBS):
        buckets.append("To-Do")

    # Events: time anchor
    if re.search(
        r"\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday"
        r"|tonight|morning|afternoon|evening|noon|\d{1,2}(am|pm)|\d{1,2}:\d{2})\b",
        lc,
    ):
        buckets.append("Events")
        if not buckets:
            buckets.append("To-Do")

    # Ideas: idea/concept language
    if re.search(r"\b(idea|concept|what if|imagine|startup|feature|build|create)\b", lc):
        buckets.append("Ideas")

    # Remember: location/credential/fact (no strong action verb)
    if re.search(r"\b(parked|password|code|pin|address|at|in|located)\b", lc):
        if "To-Do" not in buckets:
            buckets.append("Remember")

    return list(dict.fromkeys(buckets)) or ["Random"]


def _extract_time_mention(content: str, ref: datetime) -> Tuple[Optional[str], Optional[str]]:
    """
    Quick regex-based time extraction.
    Returns (date_str YYYY-MM-DD, time_str HH:MM) — both optional.
    """
    lc = content.lower()
    date_str: Optional[str] = None
    time_str: Optional[str] = None

    # Relative day
    if "today" in lc:
        date_str = ref.strftime("%Y-%m-%d")
    elif "tomorrow" in lc:
        date_str = (ref + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        # Day name
        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        for i, d in enumerate(days):
            if d in lc:
                delta = (i - ref.weekday()) % 7
                if delta == 0:
                    delta = 7
                date_str = (ref + timedelta(days=delta)).strftime("%Y-%m-%d")
                break

    # Time: "12 pm", "3:30pm", "noon", "midnight"
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

    return date_str, time_str


# ─────────────────────────────────────────────────────────────────────────────
# Main processor
# ─────────────────────────────────────────────────────────────────────────────


class MessageProcessor:
    """Transform messages into structured multi-bucket knowledge."""

    def __init__(self, cerebras_client: CerebrasClient, reminder_service=None):
        self.cerebras = cerebras_client
        self.reminder_service = reminder_service

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
    ) -> Dict:
        user = await self._get_user(user_phone, db)
        if not user:
            raise ValueError("User not registered")

        ref = datetime.utcnow()

        # ── Fast hints (zero tokens) ──────────────────────────────
        fast_buckets = _sniff_buckets_fast(content)
        fast_date, fast_time = _extract_time_mention(content, ref)

        # ── Single LLM call: full understanding + multi-bucket ────
        analysis = await self._full_analysis(
            content=content,
            user=user,
            db=db,
            fast_buckets=fast_buckets,
            fast_date=fast_date,
            fast_time=fast_time,
            ref=ref,
        )

        buckets: List[str] = analysis.get("buckets", fast_buckets)
        # Validate
        buckets = [b for b in buckets if b in BUCKET_NAMES]
        if not buckets:
            buckets = ["Random"]

        # ── Handle To-Do list splitting ───────────────────────────
        if "To-Do" in buckets and message_type == "text":
            todo_items = analysis.get("todo_items", [])
            if todo_items and len(todo_items) > 1:
                return await self._process_todo_batch(
                    user=user,
                    items=todo_items,
                    original_content=content,
                    analysis=analysis,
                    all_buckets=buckets,
                    db=db,
                )

        # ── Persist (one row, multiple bucket tags) ───────────────
        primary_bucket = buckets[0]
        category = await self._get_or_create_category(
            user_id=user.id,
            name=primary_bucket,
            auto_description=INTENT_BUCKETS.get(primary_bucket, ""),
            db=db,
        )

        due_date: Optional[str] = analysis.get("due_date")
        if not due_date and "To-Do" in buckets:
            due_date = _today_str()

        events: List[Dict] = analysis.get("events", [])

        # Build rich tags
        tags = {
            "keywords":        analysis.get("keywords", []),
            "entities":        analysis.get("entities", {}),
            "concepts":        analysis.get("concepts", []),
            "actionables":     analysis.get("actionables", []),
            "sentiment":       analysis.get("sentiment", "neutral"),
            "priority":        analysis.get("priority", "normal"),
            "time_reference":  analysis.get("time_reference", "none"),
            "event_time":      analysis.get("event_time"),
            "all_buckets":     buckets,           # ← multi-bucket
            "primary_bucket":  primary_bucket,
            "due_date":        due_date,
            "events":          events,
        }

        message = Message(
            user_id=user.id,
            category_id=category.id,
            content=content,
            message_type=MessageType(message_type),
            media_url=media_url,
            summary=analysis.get("essence", content[:100]),
            tags=tags,
            created_at=ref,
        )

        db.add(message)
        await db.commit()
        await db.refresh(message)

        result = {
            "message_id":  message.id,
            "category":    primary_bucket,
            "all_buckets": buckets,
            "tags":        analysis.get("keywords", []),
            "essence":     analysis.get("essence", ""),
            "connections": analysis.get("related_concepts", []),
            "due_date":    due_date,
            "events":      events,
            "priority":    analysis.get("priority", "normal"),
        }

        # THEN handle reminder
        if ("To-Do" in buckets or "Events" in buckets) and self.reminder_service:
            has_time = analysis.get("event_time") and analysis.get("due_date")
            if has_time:
                reminder = await self.reminder_service.create(
                    user=user,
                    content=content,
                    analysis=analysis,
                    message_id=message.id,
                    db=db,
                )
                if reminder:
                    result["reminder_id"] = reminder.id
                    result["remind_at"] = reminder.remind_at.isoformat()

        await self._save_embedding(message.id, content, analysis, db)
        return result

    # ──────────────────────────────────────────────────────────────
    # Single LLM analysis call (replaces 3-4 separate calls)
    # ──────────────────────────────────────────────────────────────

    async def _full_analysis(
        self,
        content: str,
        user: User,
        db: AsyncSession,
        fast_buckets: List[str],
        fast_date: Optional[str],
        fast_time: Optional[str],
        ref: datetime,
    ) -> Dict:
        recent_topics = await self._get_recent_topics(user.id, db)
        ref_str = ref.strftime("%Y-%m-%d (%A, %d %B %Y)")
        tomorrow = (ref + timedelta(days=1)).strftime("%Y-%m-%d")
        next_days = {
            d: (ref + timedelta(days=i)).strftime("%Y-%m-%d")
            for i, d in enumerate(
                ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"], 1
            )
        }

        bucket_defs = "\n".join(
            f'  "{name}": {desc}' for name, desc in INTENT_BUCKETS.items()
        )

        prompt = f"""You are analyzing a personal note saved by {user.name} ({user.occupation}).
TODAY: {ref_str}
TOMORROW: {tomorrow}
RECENT TOPICS: {", ".join(recent_topics[:10])}

NOTE: "{content}"

────────────────────────────────────────────────────────
TASK: Return a rich JSON analysis. Every field is required.
────────────────────────────────────────────────────────

BUCKET DEFINITIONS (choose 1-3 that apply):
{bucket_defs}

MULTI-BUCKET RULE:
  "call Kailash at 12 pm tomorrow"  → ["To-Do", "Events"]   (it's both an action AND a scheduled event)
  "buy milk"                        → ["To-Do"]
  "parked at C3"                    → ["Remember"]
  "weight 74kg"                     → ["Track"]
  "mom visiting 10th March"         → ["Events"]
  "startup idea: X"                 → ["Ideas"]
  "heyy"                            → ["Random"]
  "dentist at 3pm friday, also call john"  → ["To-Do", "Events"]

FAST HINTS (may be wrong, use as input only):
  fast_buckets = {fast_buckets}
  fast_date    = {fast_date}
  fast_time    = {fast_time}

DATE RESOLUTION:
  today    = {ref.strftime("%Y-%m-%d")}
  tomorrow = {tomorrow}
  Day names resolve to the NEXT upcoming occurrence from today.

PRIORITY DETECTION:
  urgent/asap/important/critical → "high"
  normal note                    → "normal"
  someday/eventually             → "low"

Return ONLY this JSON (no markdown):
{{
  "buckets": ["Primary", "Secondary?"],
  "essence": "one clear sentence capturing core meaning",
  "keywords": ["3-6 searchable keywords"],
  "entities": {{
    "people":        [],
    "places":        [],
    "organizations": [],
    "products":      [],
    "times":         [],
    "phone_numbers": [],
    "emails":        [],
    "urls":          []
  }},
  "concepts":       ["abstract themes"],
  "actionables":    ["specific action items"],
  "sentiment":      "neutral | positive | excited | urgent | anxious | contemplative",
  "priority":       "high | normal | low",
  "time_reference": "now | today | tomorrow | this_week | future | none",
  "event_time":     "HH:MM or null",
  "due_date":       "YYYY-MM-DD or null",
  "events": [
    {{"date": "YYYY-MM-DD", "time": "HH:MM or null", "label": "short human title"}}
  ],
  "todo_items": [
    {{"task": "clean task", "due_date": "YYYY-MM-DD", "time": "HH:MM or null"}}
  ],
  "related_concepts": ["from user's recent topics this connects to"]
}}"""

        response = await self.cerebras.chat(prompt, max_tokens=1500)

        # Defaults
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

        # Ensure entities sub-keys
        ent = response["entities"]
        if not isinstance(ent, dict):
            response["entities"] = {}
        for k in ("people", "places", "organizations", "products", "times",
                  "phone_numbers", "emails", "urls"):
            response["entities"].setdefault(k, [])

        # Validate due_date format
        dd = response.get("due_date")
        if dd and not _DATE_RE.match(str(dd)):
            response["due_date"] = None

        # Validate events
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
    # Multi To-Do splitting
    # ──────────────────────────────────────────────────────────────

    async def _process_todo_batch(
        self,
        user: User,
        items: List[Dict],
        original_content: str,
        analysis: Dict,
        all_buckets: List[str],
        db: AsyncSession,
    ) -> Dict:
        category = await self._get_or_create_category(
            user_id=user.id,
            name="To-Do",
            auto_description=INTENT_BUCKETS["To-Do"],
            db=db,
        )

        saved_ids = []
        saved_items = []  # track (message_id, task, evt_time, due_date) for post-commit work
        ref = datetime.utcnow()

        for item in items:
            task     = str(item.get("task", "")).strip()
            due_date = item.get("due_date") or _today_str()
            evt_time = item.get("time")

            if not task:
                continue

            if not _DATE_RE.match(str(due_date)):
                due_date = _today_str()

            # Does this task item also qualify as an Event?
            item_buckets = ["To-Do"]
            if evt_time or item.get("date"):
                item_buckets.append("Events")

            message = Message(
                user_id=user.id,
                category_id=category.id,
                content=task,
                message_type=MessageType("text"),
                media_url=None,
                summary=task[:100],
                tags={
                    "keywords":       analysis.get("keywords", []),
                    "entities":       analysis.get("entities", {}),
                    "concepts":       analysis.get("concepts", []),
                    "actionables":    [task],
                    "sentiment":      "neutral",
                    "priority":       analysis.get("priority", "normal"),
                    "time_reference": "none",
                    "event_time":     evt_time,
                    "all_buckets":    item_buckets,
                    "primary_bucket": "To-Do",
                    "due_date":       due_date,
                    "events":         (
                        [{"date": due_date, "time": evt_time, "label": task[:40]}]
                        if "Events" in item_buckets else []
                    ),
                    "split_from":     original_content[:200],
                },
                created_at=ref,
            )
            db.add(message)
            await db.flush()  # get message.id without committing

            saved_ids.append(message.id)
            saved_items.append({
                "id":       message.id,
                "task":     task,
                "evt_time": evt_time,
                "due_date": due_date,
            })

        # Single commit for all messages
        await db.commit()

        # Post-commit: embeddings + reminders (safe now that IDs are stable)
        for saved in saved_items:
            # Embeddings
            await self._save_embedding(saved["id"], saved["task"], analysis, db)

            # Reminders — only if has a time and reminder_service is wired
            if self.reminder_service and saved["evt_time"]:
                item_analysis = {
                    **analysis,
                    "event_time": saved["evt_time"],
                    "due_date":   saved["due_date"],
                }
                try:
                    await self.reminder_service.create(
                        user=user,
                        content=saved["task"],
                        analysis=item_analysis,
                        message_id=saved["id"],
                        db=db,
                    )
                except Exception as e:
                    print(f"⚠ Reminder creation failed for task '{saved['task']}': {e}")

        return {
            "message_id":  saved_ids[0] if saved_ids else None,
            "split_count": len(saved_ids),
            "message_ids": saved_ids,
            "category":    "To-Do",
            "all_buckets": ["To-Do"],
            "tags":        analysis.get("keywords", []),
            "essence":     f"Saved {len(saved_ids)} tasks: " + ", ".join(
                i["task"] for i in items[:3] if i.get("task")
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
        self,
        user_id: int,
        name: str,
        auto_description: str,
        db: AsyncSession,
    ) -> Category:
        result = await db.execute(
            select(Category).where(
                Category.user_id == user_id,
                Category.name == name,
            )
        )
        category = result.scalar_one_or_none()
        if not category:
            category = Category(
                user_id=user_id,
                name=name,
                description=auto_description,
            )
            db.add(category)
            await db.flush()
        return category

    async def _get_user(self, phone: str, db: AsyncSession) -> Optional[User]:
        result = await db.execute(select(User).where(User.phone_number == phone))
        return result.scalar_one_or_none()

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

    async def _save_embedding(
        self,
        message_id: int,
        content: str,
        analysis: Dict,
        db: AsyncSession,
    ):
        try:
            from services.embedding_service import embedding_service

            # Enrich the text before embedding for better retrieval
            people    = analysis.get("entities", {}).get("people", [])
            keywords  = analysis.get("keywords", [])
            essence   = analysis.get("essence", "")
            buckets   = analysis.get("buckets", [])
            actionables = analysis.get("actionables", [])

            enriched = " ".join(filter(None, [
                content,
                essence,
                " ".join(keywords),
                " ".join(people),
                " ".join(actionables),
                " ".join(buckets),
            ]))

            embedding = await embedding_service.aembed(enriched.strip(), task_type="RETRIEVAL_DOCUMENT")
            await db.execute(
                update(Message)
                .where(Message.id == message_id)
                .values(embedding=embedding)
            )
            await db.commit()
        except Exception as e:
            print(f"⚠ Embedding failed (non-critical): {e}")
