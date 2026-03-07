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
        4. Generate rich, searchable metadata
        5. Build connections to existing knowledge
        6. Extract structured calendar events (NEW)
        """
        
        # Get user
        user = await self._get_user(user_phone, db)
        if not user:
            raise ValueError("User not registered")
        
        # ===== STEP 1: DEEP UNDERSTANDING =====
        understanding = await self._deep_understand(content, user, db)

        # Safety net if LLM response was malformed
        if "suggested_category" not in understanding:
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
            db=db
        )

        # ===== STEP 3: EXTRACT CALENDAR EVENTS (parallel-safe) =====
        # Only attempt for text messages that may contain dates/events
        events: List[Dict] = []
        if message_type == "text":
            events = await self._extract_events(content, datetime.utcnow())
        
        # ===== STEP 4: SAVE WITH RICH METADATA =====
        message = Message(
            user_id=user.id,
            category_id=category.id,
            content=content,
            message_type=MessageType(message_type),
            media_url=media_url,
            
            # Rich metadata for search
            summary=understanding["essence"],
            tags={
                "keywords":       understanding["keywords"],
                "entities":       understanding["entities"],
                "concepts":       understanding["concepts"],
                "actionables":    understanding["actionables"],
                "sentiment":      understanding["sentiment"],
                "time_reference": understanding["time_reference"],
                # Calendar events — consumed by the Visual calendar on the frontend
                "events":         events,
            },
            created_at=datetime.utcnow()
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
            "events":      events,   # useful for callers to know what was detected
        }
    
    # ─────────────────────────────────────────────────────────────
    # NEW: Calendar event extraction
    # ─────────────────────────────────────────────────────────────

    async def _extract_events(
        self,
        content: str,
        ref_date: datetime,
    ) -> List[Dict]:
        """
        Use Cerebras to extract ALL structured calendar events from content.

        Handles:
          - Explicit dates:   "24 feb", "10,11 march"
          - Relative dates:   "after 3 days", "next Monday", "this weekend"
          - Multi-date spans: "20,21 march - shikhar"  →  two entries
          - Smart labels:     raw "archi brother" → "Archi's brother's wedding"

        Returns a list of {date, label} dicts, e.g.:
          [
            {"date": "2025-02-24", "label": "Chidiya's wedding"},
            {"date": "2025-03-10", "label": "Archi's brother's wedding"},
            {"date": "2025-03-11", "label": "Archi's brother's wedding"},
            {"date": "2025-03-20", "label": "Shikhar's wedding"},
            {"date": "2025-03-21", "label": "Shikhar's wedding"},
          ]

        Returns [] if no events found or if LLM call fails.
        """
        prompt = f"""Extract ALL calendar events and dates from the message below.
Reference date (when this was saved): {ref_date.strftime('%Y-%m-%d')}

Message: "{content}"

Instructions:
- Resolve relative dates using the reference date above (e.g. "after 3 days" = ref + 3 days)
- For comma-separated days like "10,11 march", emit one entry PER day
- Labels must be short, human-friendly, and deduced from context
  Good: "Archi's brother's wedding", "Flight to Bangalore", "Chidiya's marriage"
  Bad:  "archi brother", "marriage to attend", the raw message text
- If the same event spans multiple days (e.g. "20,21 march - shikhar"), repeat with same label
- If no events or dates are found, return an empty array
- Return ONLY a valid JSON array, no explanation, no markdown fences

Expected format:
[
  {{"date": "YYYY-MM-DD", "label": "short human-friendly title"}},
  ...
]"""

        try:
            raw = await self.cerebras.chat(prompt, max_tokens=512)

            # cerebras.chat may return a dict (parsed JSON) or a string
            if isinstance(raw, list):
                events = raw
            elif isinstance(raw, dict):
                # Sometimes the LLM wraps the array: {"events": [...]}
                events = raw.get("events", raw.get("dates", []))
            else:
                # Strip markdown fences if present and parse
                cleaned = re.sub(r"```(?:json)?|```", "", str(raw)).strip()
                events = json.loads(cleaned)

            # Validate each entry
            validated = []
            for e in events:
                if (
                    isinstance(e, dict)
                    and "date" in e
                    and "label" in e
                    and isinstance(e["date"], str)
                    and len(e["date"]) == 10          # YYYY-MM-DD
                    and isinstance(e["label"], str)
                    and e["label"].strip()
                ):
                    validated.append({
                        "date":  e["date"],
                        "label": e["label"].strip(),
                    })

            return validated

        except Exception as ex:
            # Non-critical — never block the main save flow
            print(f"⚠ Event extraction failed (non-critical): {ex}")
            return []

    # ─────────────────────────────────────────────────────────────
    # Existing methods below — unchanged
    # ─────────────────────────────────────────────────────────────

    async def _deep_understand(
        self,
        content: str,
        user: User,
        db: AsyncSession
    ) -> Dict:
        """
        Use AI to deeply understand the message:
        - What is this really about? (essence)
        - What does the user intend? (intent)
        - What are the key concepts?
        - Are there actionable items?
        - How does this relate to their existing knowledge?
        """
        
        # Get user's recent context for better understanding
        recent_categories = await self._get_user_categories(user.id, db)
        recent_topics = await self._get_recent_topics(user.id, db)
        
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
    
    async def _intelligent_categorize(
        self,
        content: str,
        understanding: Dict,
        user: User,
        db: AsyncSession
    ) -> Category:
        """
        Find the PERFECT category - not generic buckets.
        
        Rules:
        1. Use existing category if content clearly fits
        2. Create NEW specific category if this is a new theme
        3. Never use vague categories like "Notes" or "General"
        4. Category names should be MEANINGFUL and SEARCHABLE
        """
        
        suggested_category = understanding.get("suggested_category", "General Notes")
        intent = understanding.get("intent", "note")
        
        # Get all user's categories
        existing_categories = await self._get_all_user_categories(user.id, db)
        
        if not existing_categories:
            # First message - create meaningful category
            category_name = suggested_category
        else:
            # Find best match or create new
            category_name = await self._find_best_category_match(
                suggested=suggested_category,
                existing=[c.name for c in existing_categories],
                content=content,
                understanding=understanding
            )
        
        # Get or create category
        category = await self._get_or_create_category(
            user_id=user.id,
            name=category_name,
            auto_description=f"{intent.capitalize()} - {understanding['essence'][:100]}",
            db=db
        )
        
        return category
    
    async def _find_best_category_match(
        self,
        suggested: str,
        existing: List[str],
        content: str,
        understanding: Dict
    ) -> str:
        """Use AI to find best category match or suggest new one"""
        
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
        db: AsyncSession
    ) -> Category:
        """Get existing or create new category"""
        
        result = await db.execute(
            select(Category).where(
                Category.user_id == user_id,
                Category.name == name
            )
        )
        category = result.scalar_one_or_none()
        
        if not category:
            category = Category(
                user_id=user_id,
                name=name,
                description=auto_description
            )
            db.add(category)
            await db.flush()
        
        return category
    
    async def _get_user(self, phone: str, db: AsyncSession) -> Optional[User]:
        result = await db.execute(select(User).where(User.phone_number == phone))
        return result.scalar_one_or_none()
    
    async def _get_user_categories(self, user_id: int, db: AsyncSession) -> List[str]:
        """Get user's category names for context"""
        result = await db.execute(
            select(Category.name).where(Category.user_id == user_id).limit(20)
        )
        return [row[0] for row in result.all()]
    
    async def _get_recent_topics(self, user_id: int, db: AsyncSession) -> List[str]:
        """Get recent topics/keywords for context"""
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
        """Get all categories for a user"""
        result = await db.execute(
            select(Category).where(Category.user_id == user_id)
        )
        return list(result.scalars().all())
    
    async def _save_embedding(
        self,
        message_id: int,
        content: str,
        understanding: Dict,
        db: AsyncSession
    ):
        """Generate and store vector embedding for semantic search"""
        try:
            from services.embedding_service import embedding_service
            
            # Combine content + essence for richer embedding
            text_to_embed = f"{content} {understanding.get('essence', '')}".strip()
            
            embedding = embedding_service.embed(text_to_embed)
            
            await db.execute(
                update(Message)
                .where(Message.id == message_id)
                .values(embedding=embedding)
            )
            await db.commit()
            
        except Exception as e:
            print(f"⚠ Embedding failed (non-critical): {e}")
            # Never block the main save flow
