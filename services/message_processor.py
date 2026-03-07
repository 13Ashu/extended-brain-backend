"""
Intelligent Message Processor
Transforms raw thoughts into organized, searchable knowledge

Core Philosophy:
- Understand INTENT, not just content
- INTENT-BASED bucketing — not topic-based
- Build CONNECTIONS between ideas
- Generate RICH metadata for powerful search
"""

from typing import Dict, Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from datetime import datetime, timedelta
import json
import re

from database import User, Message, Category
from cerebras_client import CerebrasClient
from models import MessageType


# ─────────────────────────────────────────────────────────────────────────────
# INTENT BUCKETS — the only categories that should ever be created
# Topic-based buckets (Parking, Cosmetics, Office) are BANNED.
# ─────────────────────────────────────────────────────────────────────────────

INTENT_BUCKETS = {
    "Remember": (
        "User wants to remember a fact, location, or piece of information for later. "
        "Examples: 'parked at C3', 'nailcutter in 3rd drawer', 'wifi password is 1234', "
        "'mom's birthday is June 3rd', 'meeting room is 4B'."
    ),
    "To-Do": (
        "User has a task, action item, or reminder for themselves. "
        "Examples: 'buy milk', 'call dentist', 'submit report by Friday', "
        "'remind me to water plants', 'need to renew passport'."
    ),
    "Ideas": (
        "User is capturing a new idea, thought, concept, or creative spark. "
        "Examples: 'what if we built X', 'startup idea: Y', 'feature idea for the app', "
        "'interesting concept: Z', 'could try this approach for...'."
    ),
    "Track": (
        "User wants to log or monitor something over time — health, habits, progress, mood. "
        "Examples: 'weight today 74kg', 'ran 5km', 'mood: anxious today', "
        "'slept 6 hours', 'drank 2L water', 'steps: 8000'."
    ),
    "Events": (
        "User is noting a time-anchored event, appointment, or plan. "
        "Examples: 'mom visiting 10th march', 'dentist appointment Friday 3pm', "
        "'wedding next Saturday', 'flight to Delhi on 20th'."
    ),
    "Random": (
        "Casual, venting, conversational, or unclear intent — not meant to be retrieved. "
        "Examples: 'heyy', 'lol okay', 'ugh today was rough', 'testing 123', "
        "'hi', 'hello', single word messages, emotional venting with no actionable."
    ),
}

BUCKET_NAMES = list(INTENT_BUCKETS.keys())

# ─────────────────────────────────────────────────────────────────────────────
# Pure helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_relative_date(time_ref: str, ref_date: datetime) -> Optional[str]:
    d: Optional[datetime] = None
    if time_ref in ("now", "today"):
        d = ref_date
    elif time_ref == "this_week":
        d = ref_date - timedelta(days=ref_date.weekday())
    elif time_ref == "future":
        d = ref_date + timedelta(days=1)
    return d.strftime("%Y-%m-%d") if d else None


def _synthesize_label(content: str, people: List[str], concepts: List[str]) -> str:
    VISIT_CONCEPTS    = {"visitation", "visit", "arrival", "coming"}
    WEDDING_CONCEPTS  = {"wedding", "marriage", "ceremony", "engagement"}
    MEETING_CONCEPTS  = {"meeting", "appointment", "standup", "sync"}
    CALL_CONCEPTS     = {"call", "phone", "video call"}
    DEADLINE_CONCEPTS = {"deadline", "due", "submission", "submit"}

    lc = {c.lower() for c in concepts}

    if people:
        name = people[0].strip().capitalize()
        if lc & WEDDING_CONCEPTS:  return f"{name}'s wedding"
        if lc & VISIT_CONCEPTS:    return f"{name}'s visit"
        if lc & MEETING_CONCEPTS:  return f"Meeting with {name}"
        if lc & CALL_CONCEPTS:     return f"Call with {name}"
        return f"{name}'s event"

    if lc & DEADLINE_CONCEPTS: return "Deadline"
    if concepts:               return concepts[0].strip().capitalize()
    return content[:40].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Main processor
# ─────────────────────────────────────────────────────────────────────────────

class MessageProcessor:
    """Transform messages into structured knowledge"""

    def __init__(self, cerebras_client: CerebrasClient):
        self.cerebras = cerebras_client

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

        # ── STEP 1: DEEP UNDERSTANDING ────────────────────────────
        understanding = await self._deep_understand(content, user, db)

        understanding.setdefault("essence", content[:100])
        understanding.setdefault("keywords", [])
        understanding.setdefault("concepts", [])
        understanding.setdefault("entities", {})
        understanding.setdefault("actionables", [])
        understanding.setdefault("sentiment", "neutral")
        understanding.setdefault("time_reference", "none")
        understanding.setdefault("related_concepts", [])
        understanding.setdefault("intent", "Random")
        understanding.setdefault("intent_bucket", "Random")

        # ── STEP 2: ASSIGN INTENT BUCKET ─────────────────────────
        bucket_name = await self._assign_intent_bucket(content, understanding)

        # ── STEP 3: GET OR CREATE THE BUCKET CATEGORY ────────────
        category = await self._get_or_create_category(
            user_id=(await self._get_user(user_phone, db)).id,
            name=bucket_name,
            auto_description=INTENT_BUCKETS.get(bucket_name, ""),
            db=db,
        )

        # ── STEP 4: EXTRACT CALENDAR EVENTS ──────────────────────
        events: List[Dict] = []
        if message_type == "text":
            events = await self._extract_events(
                content=content,
                ref_date=datetime.utcnow(),
                understanding=understanding,
            )

        # ── STEP 4b: EXTRACT DUE DATE FOR TO-DOS ─────────────────
        due_date: Optional[str] = None
        if bucket_name == "To-Do" and message_type == "text":
            due_date = await self._extract_due_date(
                content=content,
                ref_date=datetime.utcnow(),
                understanding=understanding,
            )

        # ── STEP 5: SAVE WITH RICH METADATA ──────────────────────
        message = Message(
            user_id=user.id,
            category_id=category.id,
            content=content,
            message_type=MessageType(message_type),
            media_url=media_url,
            summary=understanding["essence"],
            tags={
                "keywords":       understanding["keywords"],
                "entities":       understanding["entities"],
                "concepts":       understanding["concepts"],
                "actionables":    understanding["actionables"],
                "sentiment":      understanding["sentiment"],
                "time_reference": understanding["time_reference"],
                "intent_bucket":  bucket_name,
                "due_date":       due_date,   # YYYY-MM-DD or null
                "events":         events,
            },
            created_at=datetime.utcnow(),
        )

        db.add(message)
        await db.commit()
        await db.refresh(message)

        # ── STEP 6: GENERATE & SAVE EMBEDDING ────────────────────
        await self._save_embedding(message.id, content, understanding, db)

        return {
            "message_id":    message.id,
            "category":      category.name,
            "intent_bucket": bucket_name,
            "tags":          understanding["keywords"],
            "essence":       understanding["essence"],
            "connections":   understanding["related_concepts"],
            "due_date":      due_date,
            "events":        events,
        }

    # ─────────────────────────────────────────────────────────────
    # Intent bucket assignment  ← THE CORE CHANGE
    # ─────────────────────────────────────────────────────────────

    async def _assign_intent_bucket(self, content: str, understanding: Dict) -> str:
        """
        Assign one of the 6 fixed intent buckets.
        Topic-based categories are NEVER created.
        """

        bucket_descriptions = "\n".join(
            f'  "{name}": {desc}' for name, desc in INTENT_BUCKETS.items()
        )

        prompt = f"""You are classifying a personal note by WHY the user saved it — their INTENT.

AVAILABLE BUCKETS (these are the ONLY valid options):
{bucket_descriptions}

NOTE: "{content}"

ANALYSIS CONTEXT:
- Essence    : {understanding.get('essence', '')}
- Actionables: {understanding.get('actionables', [])}
- Time ref   : {understanding.get('time_reference', 'none')}
- Entities   : {understanding.get('entities', {})}

RULES:
1. Pick exactly ONE bucket from the list above.
2. Ask yourself: WHY did the user write this? What will they do with it?
   - To recall it later?        → Remember
   - To act on it?              → To-Do
   - To explore an idea?        → Ideas
   - To log progress/habit?     → Track
   - To note a future event?    → Events
   - Just venting / casual?     → Random
3. NEVER create a topic-based name like "Parking", "Cosmetics", "Food".
4. "parked at C3" → Remember  (they want to recall WHERE they parked)
5. "nailcutter in 3rd drawer" → Remember
6. "buy milk" → To-Do
7. "heyy" → Random
8. "startup idea: X" → Ideas
9. "weight 74kg today" → Track
10. "mom visiting 10th march" → Events

Return ONLY this JSON:
{{"bucket": "one of the 6 bucket names above"}}"""

        response = await self.cerebras.chat(prompt)
        bucket = response.get("bucket", "Random")

        # Safety: ensure it's a valid bucket
        if bucket not in BUCKET_NAMES:
            bucket = "Random"

        return bucket

    # ─────────────────────────────────────────────────────────────
    # Due date extraction for To-Do messages
    # ─────────────────────────────────────────────────────────────

    async def _extract_due_date(
        self,
        content: str,
        ref_date: datetime,
        understanding: Dict,
    ) -> str:
        """
        Extract or infer a due date for a To-Do item.
        Always returns a YYYY-MM-DD string — never None.

        - Explicit date/day in content → LLM resolves it
        - No temporal signal at all    → today (skip LLM entirely)
        """
        time_ref = understanding.get("time_reference", "none")
        today    = ref_date.strftime("%Y-%m-%d")

        # Fast-path: no temporal signal → always today, skip token call
        if time_ref == "none":
            return today

        prompt = f"""You extract the due date from a to-do note.

Reference date (today): {ref_date.strftime('%Y-%m-%d')} ({ref_date.strftime('%A, %d %B %Y')})

To-do: "{content}"

━━━ DATE RESOLUTION ━━━
  "today"      → {today}
  "tomorrow"   → {(ref_date + timedelta(days=1)).strftime('%Y-%m-%d')}
  "by Friday"  → next Friday from ref date
  "next week"  → Monday of next week
  "by 20th"    → {ref_date.year}-{ref_date.month:02d}-20 (next month if already past)

IMPORTANT: Only use a future date if the note EXPLICITLY mentions one.
If unclear, return today: {today}

Return ONLY this JSON — no markdown, no explanation:
{{"due_date": "YYYY-MM-DD"}}"""

        try:
            response = await self.cerebras.chat(prompt, max_tokens=60)
            due = response.get("due_date")
            if due and re.match(r"^\d{4}-\d{2}-\d{2}$", str(due)):
                return due
        except Exception as e:
            print(f"⚠ Due date extraction failed: {e}")

        return today

    # ─────────────────────────────────────────────────────────────
    # Calendar event extraction (unchanged)
    # ─────────────────────────────────────────────────────────────

    async def _extract_events(
        self,
        content: str,
        ref_date: datetime,
        understanding: Dict,
    ) -> List[Dict]:
        time_ref    = understanding.get("time_reference", "none")
        people      = understanding.get("entities", {}).get("people", [])
        actionables = understanding.get("actionables", [])
        concepts    = understanding.get("concepts", [])

        if time_ref == "none" and not actionables and not people:
            return []

        prompt = f"""You extract calendar events from personal notes.
CRITICAL: A time anchor IS present in this message. You MUST return at least one event — never return [].

Reference date: {ref_date.strftime('%Y-%m-%d')} ({ref_date.strftime('%A, %d %B %Y')})

Message: "{content}"

Already known from prior analysis:
- Time reference : {time_ref}
- People         : {people}
- Actionables    : {actionables}
- Concepts       : {concepts}
- Essence        : {understanding.get('essence', '')}

━━━ DATE RESOLUTION ━━━
Resolve every date/time expression to YYYY-MM-DD:
  "today"         → {ref_date.strftime('%Y-%m-%d')}
  "tomorrow"      → {(ref_date + timedelta(days=1)).strftime('%Y-%m-%d')}
  "day after"     → {(ref_date + timedelta(days=2)).strftime('%Y-%m-%d')}
  "after N days"  → ref + N days
  "next Monday"   → the Monday after ref date
  "this weekend"  → the coming Saturday from ref date
  "next week"     → Monday of next week
  "24 feb"        → {ref_date.year}-02-24  (next year if already past)
  "10,11 march"   → TWO entries: {ref_date.year}-03-10 AND {ref_date.year}-03-11

━━━ LABEL RULES ━━━
Short, human, inferred from full context:
  "mom is coming tomorrow"   → "Mom's visit"
  "flight to delhi next fri" → "Flight to Delhi"
  "dinner with rahul sunday" → "Dinner with Rahul"

Return ONLY a valid JSON array:
[{{"date": "YYYY-MM-DD", "label": "short human title"}}]"""

        validated: List[Dict] = []

        try:
            raw = await self.cerebras.chat(prompt, max_tokens=512)

            if isinstance(raw, list):
                events = raw
            elif isinstance(raw, dict):
                events = raw.get("events", raw.get("dates", []))
            else:
                cleaned = re.sub(r"```(?:json)?|```", "", str(raw)).strip()
                parsed  = json.loads(cleaned)
                events  = parsed if isinstance(parsed, list) else [parsed]

            for e in events:
                if not isinstance(e, dict):
                    continue
                date  = e.get("date", "")
                label = e.get("label", "")
                if (
                    re.match(r"^\d{4}-\d{2}-\d{2}$", date)
                    and isinstance(label, str)
                    and label.strip()
                ):
                    validated.append({"date": date, "label": label.strip()})

        except Exception as ex:
            print(f"⚠ Event extraction (LLM) failed: {ex}")

        if not validated and time_ref != "none":
            date  = _resolve_relative_date(time_ref, ref_date)
            label = _synthesize_label(content, people, concepts)
            if date and label:
                validated.append({"date": date, "label": label})
                print(f"ℹ Event fallback used: '{label}' → {date}")

        return validated

    # ─────────────────────────────────────────────────────────────
    # Deep understanding
    # ─────────────────────────────────────────────────────────────

    async def _deep_understand(self, content: str, user: User, db: AsyncSession) -> Dict:
        recent_categories = await self._get_user_categories(user.id, db)
        recent_topics     = await self._get_recent_topics(user.id, db)

        prompt = f"""You are analyzing a personal note to extract structured metadata.

USER CONTEXT:
- Name: {user.name}
- Occupation: {user.occupation}
- Recent topics: {', '.join(recent_topics[:15])}

MESSAGE: "{content}"

Return JSON:
{{
  "essence": "One clear sentence capturing the core meaning",
  "intent": "remember/todo/idea/track/event/random",
  "keywords": ["3-5 most important searchable keywords"],
  "entities": {{"people": [], "places": [], "products": [], "companies": []}},
  "concepts": ["abstract concepts or themes"],
  "actionables": ["specific action items if any"],
  "sentiment": "neutral/positive/excited/urgent/contemplative",
  "time_reference": "now/today/this_week/future/none",
  "related_concepts": ["concepts from user's existing knowledge this relates to"]
}}"""

        response = await self.cerebras.chat(prompt, max_tokens=1200)
        return response

    # ─────────────────────────────────────────────────────────────
    # DB helpers
    # ─────────────────────────────────────────────────────────────

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

    async def _get_user_categories(self, user_id: int, db: AsyncSession) -> List[str]:
        result = await db.execute(
            select(Category.name).where(Category.user_id == user_id).limit(20)
        )
        return [row[0] for row in result.all()]

    async def _get_recent_topics(self, user_id: int, db: AsyncSession) -> List[str]:
        result = await db.execute(
            select(Message.tags)
            .where(Message.user_id == user_id)
            .order_by(Message.created_at.desc())
            .limit(20)
        )
        topics = set()
        for row in result.all():
            if row[0] and isinstance(row[0], dict):
                keywords = row[0].get("keywords", [])
                topics.update(keywords[:3])
        return list(topics)[:15]

    async def _get_all_user_categories(self, user_id: int, db: AsyncSession) -> List[Category]:
        result = await db.execute(
            select(Category).where(Category.user_id == user_id)
        )
        return list(result.scalars().all())

    # ─────────────────────────────────────────────────────────────
    # Embedding
    # ─────────────────────────────────────────────────────────────

    async def _save_embedding(
        self,
        message_id: int,
        content: str,
        understanding: Dict,
        db: AsyncSession,
    ):
        try:
            from services.embedding_service import embedding_service
            text_to_embed = f"{content} {understanding.get('essence', '')}".strip()
            embedding     = embedding_service.embed(text_to_embed)
            await db.execute(
                update(Message)
                .where(Message.id == message_id)
                .values(embedding=embedding)
            )
            await db.commit()
        except Exception as e:
            print(f"⚠ Embedding failed (non-critical): {e}")
