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
from sqlalchemy import select
from datetime import datetime

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
        
        # ===== STEP 3: SAVE WITH RICH METADATA =====
        message = Message(
            user_id=user.id,
            category_id=category.id,
            content=content,
            message_type=MessageType(message_type),
            media_url=media_url,
            
            # Rich metadata for search
            summary=understanding["essence"],
            tags={
                "keywords": understanding["keywords"],
                "entities": understanding["entities"],
                "concepts": understanding["concepts"],
                "actionables": understanding["actionables"],
                "sentiment": understanding["sentiment"],
                "time_reference": understanding["time_reference"],
            },
            created_at=datetime.utcnow()
        )
        
        db.add(message)
        await db.commit()
        await db.refresh(message)

        # ===== STEP 4: GENERATE & SAVE EMBEDDING =====
        await self._save_embedding(message.id, content, understanding, db)
        
        return {
            "message_id": message.id,
            "category": category.name,
            "tags": understanding["keywords"],
            "essence": understanding["essence"],
            "connections": understanding["related_concepts"]
        }
    
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
        concepts = understanding.get("concepts", [])
        
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
            from sqlalchemy import update
            
            # Combine content + essence for richer embedding
            # e.g. "chidiya marriage 24 feb" + "Wedding event for Chidiya on February 24th"
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