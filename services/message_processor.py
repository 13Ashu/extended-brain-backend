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
from datetime import datetime
import json
import re

from database import User, Message, Category
from cerebras_client import CerebrasClient
from models import MessageType


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
        2. Extract entities, concepts, and actionables
        3. Find or create the perfect category
        4. Extract structured calendar events (context-aware, reuses step 1)
        5. Save with rich metadata
        6. Generate & save embedding
        """

        user = await self._get_user(user_phone, db)
        if not user:
            raise ValueError("User not registered")

        # ===== STEP 1: DEEP UNDERSTANDING =====
        understanding = await self._deep_understand(content, user, db)

        # Safety net — if LLM response was malformed
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

        # ===== STEP 2: FIND OR CREATE CATEGORY =====
        category = await self._intelligent_categorize(
            content=content,
            understanding=understanding,
            user=user,
            db=db,
        )

        # ===== STEP 3: EXTRACT CALENDAR EVENTS =====
        # We pass `understanding` so this step reuses what step 1 already
        # figured out (time_reference, people, actionables) instead of
        # rediscovering it — making it both cheaper and smarter.
        events: List[Dict] = []
        if message_type == "text":
            events = await self._extract_events(
                content=content,
                ref_date=datetime.utcnow(),
                understanding=understanding,
            )

        # ===== STEP 4: SAVE WITH RICH METADATA =====
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

        # ===== STEP 5: GENERATE & SAVE EMBEDDING =====
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
        Extract ALL time-anchored occurrences from a message.

        Key design decisions:
        ─────────────────────
        1. Receives `understanding` from _deep_understand so we don't pay
           for a second full-analysis pass. The LLM only needs to resolve
           dates and generate labels, not re-infer context.

        2. Fast-path skip: if step 1 flagged time_reference="none" AND there
           are no people or actionables, the message is purely conceptual
           (e.g. "interesting idea about ML") — skip the LLM call entirely.

        3. Robust JSON parsing: handles raw list, wrapped dict {"events":[...]},
           string with markdown fences, or single-object responses.

        4. Strict date validation via regex (not just length check) to catch
           malformed responses like "2025-2-8" or "tomorrow".

        Handles every real-world pattern:
          Explicit dates      "24 feb", "10,11 march", "2025-03-15"
          Relative dates      "tomorrow", "next Monday", "after 3 days", "this weekend"
          Implicit timing     "mom is coming tomorrow", "flight next week"
          Multi-day spans     "10,11 march - archi" → two entries, same label
          Deadline phrasing   "submit by 20th", "due next Friday"
          Informal plans      "dinner with Rahul on Sunday"
        """

        time_ref    = understanding.get("time_reference", "none")
        people      = understanding.get("entities", {}).get("people", [])
        actionables = understanding.get("actionables", [])

        # Fast-path: no temporal signal whatsoever → skip LLM call
        if time_ref == "none" and not actionables and not people:
            return []

        prompt = f"""You extract calendar events from personal notes. Be inclusive — err on the side of extracting rather than skipping.

Reference date: {ref_date.strftime('%Y-%m-%d')} ({ref_date.strftime('%A, %d %B %Y')})

Message: "{content}"

Context already known:
- Time reference type : {time_ref}
- People mentioned    : {people}
- Actionables         : {actionables}
- Concepts            : {understanding.get('concepts', [])}
- Core meaning        : {understanding.get('essence', '')}

━━━ RULES ━━━

DATE RESOLUTION — resolve every date/time expression to YYYY-MM-DD:
  "tomorrow"      → ref + 1 day
  "day after"     → ref + 2 days
  "after N days"  → ref + N days
  "next Monday"   → the Monday that comes after ref date
  "this weekend"  → the coming Saturday from ref date
  "next week"     → Monday of next week
  "24 feb"        → {ref_date.year}-02-24  (use next year if date already passed)
  "10,11 march"   → emit one entry per day: {ref_date.year}-03-10 AND {ref_date.year}-03-11

MULTI-DAY: For "10,11 march - archi brother", emit two objects with identical labels.

LABEL GENERATION — short, human, inferred from full context (NOT raw text):
  "mom is coming tomorrow"              → "Mom's visit"
  "marriage to attend - chidiya"        → "Chidiya's wedding"
  "10,11 march - archi brother"         → "Archi's brother's wedding"
  "flight to delhi next friday"         → "Flight to Delhi"
  "call with investor next monday"      → "Investor call"
  "submit report by 20th"               → "Report deadline"

INCLUDE any time-anchored occurrence: visits, arrivals, weddings, flights, meetings,
calls, deadlines, dinners, birthdays, ceremonies — even casual informal ones.

If there is genuinely NO date or time anchor at all, return [].

━━━ OUTPUT ━━━
Return ONLY a valid JSON array. No explanation. No markdown. No wrapper object.

[{{"date": "YYYY-MM-DD", "label": "short human title"}}]"""

        try:
            raw = await self.cerebras.chat(prompt, max_tokens=512)

            # Handle all return shapes from cerebras.chat
            if isinstance(raw, list):
                events = raw
            elif isinstance(raw, dict):
                # e.g. {"events": [...]} or {"dates": [...]}
                events = raw.get("events", raw.get("dates", []))
            else:
                cleaned = re.sub(r"```(?:json)?|```", "", str(raw)).strip()
                parsed  = json.loads(cleaned)
                events  = parsed if isinstance(parsed, list) else [parsed]

            # Validate: must be {date: YYYY-MM-DD, label: non-empty string}
            validated = []
            for e in events:
                if not isinstance(e, dict):
                    continue
                date  = e.get("date", "")
                label = e.get("label", "")
                if (
                    re.match(r"^\d{4}-\d{2}-\d{2}$", date)   # strict format
                    and isinstance(label, str)
                    and label.strip()
                ):
                    validated.append({
                        "date":  date,
                        "label": label.strip(),
                    })

            return validated

        except Exception as ex:
            # Never block the main save flow
            print(f"⚠ Event extraction failed (non-critical): {ex}")
            return []

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
        prompt = f"""You are organizing a note into the user's knowledge base.

EXISTING CATEGORIES:
{', '.join(existing)}

NEW NOTE:
"{content}"

NOTE ANALYSIS:
- Intent: {understanding['intent']}
- Concepts: {', '.join(understanding['concepts'])}
- Suggested category: {suggested}

TASK: Choose the BEST category.

Rules:
1. Use existing category if note clearly fits (>70% match)
2. Create NEW specific category if this is a distinct theme
3. Be specific - avoid generic names
4. Good: "Startup Ideas", "Python Learning", "Meeting Notes - Project X"
5. Bad: "Notes", "Ideas", "Work", "Personal"

Return JSON:
{{
  "category": "chosen or new category name",
  "reason": "brief explanation",
  "is_new": true/false
}}
"""

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

            # Combine content + essence for richer embedding
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
