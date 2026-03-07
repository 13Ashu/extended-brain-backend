"""
Intelligent Message Processor
Transforms raw thoughts into organized, searchable knowledge

Core Philosophy:
- Understand INTENT, not just content
- Build CONNECTIONS between ideas
- Create CONTEXT-AWARE categories
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
# Pure helper functions (no async, no DB — used as deterministic fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_relative_date(time_ref: str, ref_date: datetime) -> Optional[str]:
    """
    Map a time_reference value (produced by _deep_understand) to a concrete
    YYYY-MM-DD string. Used only when the LLM extraction returns [] despite
    a clear time anchor being present.

      "now" / "today"  → today
      "this_week"      → Monday of the current week
      "future"         → tomorrow  (safest assumption for a vague future ref)
    """
    d: Optional[datetime] = None

    if time_ref in ("now", "today"):
        d = ref_date
    elif time_ref == "this_week":
        d = ref_date - timedelta(days=ref_date.weekday())
    elif time_ref == "future":
        d = ref_date + timedelta(days=1)

    return d.strftime("%Y-%m-%d") if d else None


def _synthesize_label(content: str, people: List[str], concepts: List[str]) -> str:
    """
    Build a best-effort human label purely from already-extracted metadata.
    Called only when the LLM extraction returns nothing.

    Priority: person + concept combo → lone concept → truncated raw content.
    """
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
        """
        Deep processing pipeline:
        1. Understand the essence and intent
        2. Find or create the perfect category
        3. Extract structured calendar events (reuses step 1 context)
        4. Save with rich metadata
        5. Generate & save embedding
        """

        user = await self._get_user(user_phone, db)
        if not user:
            raise ValueError("User not registered")

        # ── STEP 1: DEEP UNDERSTANDING ────────────────────────────
        understanding = await self._deep_understand(content, user, db)

        # Safety net — guard against malformed LLM responses
        understanding.setdefault("suggested_category", "General Notes")
        understanding.setdefault("essence", content[:100])
        understanding.setdefault("keywords", [])
        understanding.setdefault("concepts", [])
        understanding.setdefault("entities", {})
        understanding.setdefault("actionables", [])
        understanding.setdefault("sentiment", "neutral")
        understanding.setdefault("time_reference", "none")
        understanding.setdefault("related_concepts", [])
        understanding.setdefault("intent", "note")

        # ── STEP 2: FIND OR CREATE CATEGORY ──────────────────────
        category = await self._intelligent_categorize(
            content=content,
            understanding=understanding,
            user=user,
            db=db,
        )

        # ── STEP 3: EXTRACT CALENDAR EVENTS ──────────────────────
        # Pass `understanding` so this step reuses what step 1 already knows
        # (time_reference, people, actionables) rather than rediscovering it.
        events: List[Dict] = []
        if message_type == "text":
            events = await self._extract_events(
                content=content,
                ref_date=datetime.utcnow(),
                understanding=understanding,
            )

        # ── STEP 4: SAVE WITH RICH METADATA ──────────────────────
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
                # Consumed by the Visual calendar on the frontend
                "events":         events,
            },
            created_at=datetime.utcnow(),
        )

        db.add(message)
        await db.commit()
        await db.refresh(message)

        # ── STEP 5: GENERATE & SAVE EMBEDDING ────────────────────
        await self._save_embedding(message.id, content, understanding, db)

        return {
            "message_id":  message.id,
            "category":    category.name,
            "tags":        understanding["keywords"],
            "essence":     understanding["essence"],
            "connections": understanding["related_concepts"],
            "events":      events,
        }

    # ─────────────────────────────────────────────────────────────
    # Calendar event extraction
    # ─────────────────────────────────────────────────────────────

    async def _extract_events(
        self,
        content: str,
        ref_date: datetime,
        understanding: Dict,
    ) -> List[Dict]:
        """
        Two-layer event extraction — never misses an obvious case.

        Layer 1 — LLM:
          Handles complex patterns: multi-date spans, wedding contexts,
          "flight next Friday", etc. Receives `understanding` so it has
          full context without a redundant analysis pass.

        Layer 2 — Deterministic fallback:
          If the LLM still returns [] despite a confirmed time anchor,
          `_resolve_relative_date` + `_synthesize_label` synthesize an
          event purely from the understanding dict — zero extra tokens.
          This catches cases like "mom is coming tomorrow" where the LLM
          is sometimes too conservative.

        Fast-path skip:
          time_reference="none" AND no people AND no actionables → the
          message is purely conceptual → skip entirely, save the token call.
        """

        time_ref    = understanding.get("time_reference", "none")
        people      = understanding.get("entities", {}).get("people", [])
        actionables = understanding.get("actionables", [])
        concepts    = understanding.get("concepts", [])

        # Fast-path: genuinely no temporal signal → nothing to extract
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

━━━ MULTI-DAY ━━━
For "10,11 march - archi brother" → emit two objects with the same label.

━━━ LABEL RULES ━━━
Short, human, inferred from full context — NOT the raw message text:
  "mom is coming tomorrow"        → "Mom's visit"
  "my mom is coming tomorrow"     → "Mom's visit"
  "marriage to attend - chidiya"  → "Chidiya's wedding"
  "10,11 march - archi brother"   → "Archi's brother's wedding"
  "flight to delhi next friday"   → "Flight to Delhi"
  "call with investor monday"     → "Investor call"
  "submit report by 20th"         → "Report deadline"
  "dinner with rahul sunday"      → "Dinner with Rahul"

━━━ INCLUDE ━━━
ANY time-anchored occurrence: visits, arrivals, weddings, flights, meetings,
calls, deadlines, dinners, birthdays, ceremonies — even casual informal ones.

━━━ OUTPUT ━━━
Return ONLY a valid JSON array. No explanation. No markdown. No wrapper key.
[{{"date": "YYYY-MM-DD", "label": "short human title"}}]"""

        validated: List[Dict] = []

        try:
            raw = await self.cerebras.chat(prompt, max_tokens=512)

            # Handle all return shapes cerebras.chat might produce
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

        # ── Layer 2: deterministic fallback ──────────────────────
        # If the LLM still returned nothing despite a confirmed time anchor,
        # synthesize an event from what _deep_understand already gave us.
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

    async def _deep_understand(
        self,
        content: str,
        user: User,
        db: AsyncSession,
    ) -> Dict:
        """
        Use AI to deeply understand the message:
        - What is this really about? (essence)
        - What does the user intend? (intent)
        - What are the key concepts?
        - Are there actionable items?
        - How does this relate to their existing knowledge?
        """

        recent_categories = await self._get_user_categories(user.id, db)
        recent_topics     = await self._get_recent_topics(user.id, db)

        prompt = f"""You are analyzing a thought/note from a user's personal knowledge base.

USER CONTEXT:
- Name: {user.name}
- Occupation: {user.occupation}
- Recent categories: {', '.join(recent_categories[:10])}
- Recent topics: {', '.join(recent_topics[:15])}

MESSAGE TO ANALYZE:
"{content}"

TASK: Deep analysis to make this searchable and organizable.

Return JSON with:
{{
  "essence": "One clear sentence capturing the core meaning",
  "intent": "what/why/reminder/idea/task/reference/learning",
  "keywords": ["3-5 most important searchable keywords"],
  "entities": {{"people": [], "places": [], "products": [], "companies": []}},
  "concepts": ["abstract concepts or themes"],
  "actionables": ["specific action items if any"],
  "sentiment": "neutral/positive/excited/urgent/contemplative",
  "time_reference": "now/today/this_week/future/none",
  "related_concepts": ["concepts from user's existing knowledge this relates to"],
  "suggested_category": "best category name for this"
}}

Be specific and intelligent. Think like you're organizing this for future retrieval.
"""

        response = await self.cerebras.chat(prompt, max_tokens=1200)
        return response

    # ─────────────────────────────────────────────────────────────
    # Categorisation
    # ─────────────────────────────────────────────────────────────

    async def _intelligent_categorize(
        self,
        content: str,
        understanding: Dict,
        user: User,
        db: AsyncSession,
    ) -> Category:
        suggested_category = understanding.get("suggested_category", "General Notes")
        intent             = understanding.get("intent", "note")

        existing_categories = await self._get_all_user_categories(user.id, db)

        if not existing_categories:
            category_name = suggested_category
        else:
            category_name = await self._find_best_category_match(
                suggested=suggested_category,
                existing=[c.name for c in existing_categories],
                content=content,
                understanding=understanding,
            )

        category = await self._get_or_create_category(
            user_id=user.id,
            name=category_name,
            auto_description=f"{intent.capitalize()} - {understanding['essence'][:100]}",
            db=db,
        )

        return category

    async def _find_best_category_match(
        self,
        suggested: str,
        existing: List[str],
        content: str,
        understanding: Dict,
    ) -> str:
        prompt = f"""You are organizing a note into a personal knowledge base.

EXISTING CATEGORIES (already created by this user):
{chr(10).join(f'  - {c}' for c in existing)}

NEW NOTE:
"{content}"

NOTE ANALYSIS:
- Intent   : {understanding['intent']}
- Concepts : {', '.join(understanding['concepts'])}
- People   : {understanding.get('entities', {}).get('people', [])}
- Suggested: {suggested}

TASK: Pick the single best category name.

STRICT RULES:
1. DEFAULT to an existing category. Reuse aggressively.
2. Treat semantically similar names as THE SAME — pick whichever already exists:
   "Family Visit" = "Family and Upcoming Events" = "Family and Visitation" = "Family"
   "Python Tips"  = "Python Learning" = "Python Notes" = "Python"
3. Only create a NEW category if the note covers a topic with NO overlap
   with any existing category whatsoever.
4. Never create near-duplicates. When in doubt, reuse the closest one.
5. Names must be short (1-3 words). No "and" chaining. No verbose phrases.

Return ONLY this JSON, nothing else:
{{"category": "name", "is_new": true/false}}"""

        response = await self.cerebras.chat(prompt)
        return response.get("category", suggested)

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

    # ─────────────────────────────────────────────────────────────
    # DB helpers
    # ─────────────────────────────────────────────────────────────

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
        """Generate and store vector embedding for semantic search"""
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
