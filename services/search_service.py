"""
Intelligent Search Service
Natural language search that understands INTENT, not just keywords

Search Philosophy:
- Understand what user is LOOKING for, not just matching words
- Surface RELEVANT content, not just keyword matches
- Support conversational queries
- Learn from user's search patterns
"""

from typing import List, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, and_, func
from datetime import datetime, timedelta

from database import User, Message, Category
from cerebras_client import CerebrasClient


class SearchService:
    """Intelligent search through user's knowledge base"""
    
    def __init__(self, cerebras_client: CerebrasClient):
        self.cerebras = cerebras_client
    
    async def search(
        self,
        user_phone: str,
        query: str,
        db: AsyncSession,
        limit: int = 15,
        category_filter: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Natural language search with intelligence:
        
        User can search like:
        - "what did I say about the startup?"
        - "my ideas from last week"
        - "anything about Python learning"
        - "find that article I saved yesterday"
        - "show me urgent tasks"
        """
        
        # Get user
        user = await self._get_user(user_phone, db)
        if not user:
            return []
        
        # ===== STEP 1: UNDERSTAND SEARCH INTENT =====
        search_understanding = await self._understand_search_query(query, user, db)
        
        # ===== STEP 2: BUILD INTELLIGENT SEARCH =====
        messages = await self._intelligent_search(
            user=user,
            query=query,
            understanding=search_understanding,
            db=db,
            limit=limit * 2  # Get more for ranking
        )
        
        # ===== STEP 3: RANK BY RELEVANCE =====
        ranked = await self._rank_by_relevance(
            messages=messages,
            query=query,
            understanding=search_understanding
        )
        
        # Return top results
        return ranked[:limit]
    
    async def _understand_search_query(
        self,
        query: str,
        user: User,
        db: AsyncSession
    ) -> Dict:
        """
        Deeply understand what user is searching for
        """
        
        user_categories = await self._get_user_categories(user.id, db)
        
        prompt = f"""You are helping search a personal knowledge base.

USER: {user.name} ({user.occupation})
USER'S CATEGORIES: {', '.join(user_categories)}

SEARCH QUERY:
"{query}"

TASK: Understand what they're looking for.

Return JSON:
{{
  "intent": "find_specific/browse_category/time_based/topic_explore",
  "core_concepts": ["main concepts to find"],
  "keywords": ["expanded keywords including synonyms"],
  "time_filter": "today/this_week/last_month/none",
  "category_hints": ["likely categories"],
  "entities": ["specific people/places/things mentioned"],
  "search_focus": "content/summary/tags/category/all"
}}

Think: What is the user REALLY trying to find?
"""
        
        response = await self.cerebras.chat(prompt)
        return response
    
    async def _intelligent_search(
        self,
        user: User,
        query: str,
        understanding: Dict,
        db: AsyncSession,
        limit: int
    ) -> List[tuple]:
        """
        Search with intelligence - not just keyword matching
        """
        
        # Build base query
        stmt = (
            select(Message, Category)
            .join(Category, Message.category_id == Category.id, isouter=True)
            .where(Message.user_id == user.id)
        )
        
        # Apply time filter if specified
        time_filter = understanding.get("time_filter", "none")
        if time_filter != "none":
            cutoff = self._get_time_cutoff(time_filter)
            if cutoff:
                stmt = stmt.where(Message.created_at >= cutoff)
        
        # Build search conditions
        search_conditions = []
        
        # 1. Search in content (most important)
        for keyword in understanding["keywords"][:5]:
            search_conditions.append(
                func.lower(Message.content).contains(keyword.lower())
            )
        
        # 2. Search in essence/summary
        for concept in understanding["core_concepts"]:
            search_conditions.append(
                func.lower(Message.summary).contains(concept.lower())
            )
        
        # 3. Search in tags (JSON field)
        for keyword in understanding["keywords"][:3]:
            search_conditions.append(
                func.lower(Message.tags.cast(str)).contains(keyword.lower())
            )
        
        # 4. Search in category names
        for hint in understanding.get("category_hints", []):
            search_conditions.append(
                func.lower(Category.name).contains(hint.lower())
            )
        
        # Combine with OR (match ANY condition)
        if search_conditions:
            stmt = stmt.where(or_(*search_conditions))
        
        # Order by recency
        stmt = stmt.order_by(Message.created_at.desc()).limit(limit)
        
        # Execute
        result = await db.execute(stmt)
        return result.all()
    
    async def _rank_by_relevance(
        self,
        messages: List[tuple],
        query: str,
        understanding: Dict
    ) -> List[Dict]:
        """
        Rank results by relevance - not just recency
        """
        
        scored_results = []
        
        for message, category in messages:
            score = self._calculate_relevance_score(
                message=message,
                category=category,
                query=query,
                understanding=understanding
            )
            
            scored_results.append({
                "id": message.id,
                "content": message.content,
                "essence": message.summary,
                "category": category.name if category else "Uncategorized",
                "tags": message.tags.get("keywords", []) if message.tags else [],
                "created_at": message.created_at.isoformat(),
                "relevance": score,
                "preview": self._generate_preview(message.content, understanding["keywords"])
            })
        
        # Sort by relevance score
        scored_results.sort(key=lambda x: x["relevance"], reverse=True)
        
        return scored_results
    
    def _calculate_relevance_score(
        self,
        message: Message,
        category: Optional[Category],
        query: str,
        understanding: Dict
    ) -> float:
        """
        Sophisticated relevance scoring:
        - Exact matches > partial matches
        - Content matches > tag matches
        - Recent > old (but not exclusively)
        - Intent alignment
        """
        
        score = 0.0
        content_lower = message.content.lower()
        summary_lower = (message.summary or "").lower()
        
        # === CONTENT MATCHING ===
        # Exact phrase match (highest weight)
        if query.lower() in content_lower:
            score += 10.0
        
        # Core concept matches
        for concept in understanding["core_concepts"]:
            if concept.lower() in content_lower:
                score += 5.0
            if concept.lower() in summary_lower:
                score += 3.0
        
        # Keyword matches
        for keyword in understanding["keywords"]:
            if keyword.lower() in content_lower:
                score += 2.0
            if keyword.lower() in summary_lower:
                score += 1.0
        
        # === TAG MATCHING ===
        if message.tags:
            msg_keywords = message.tags.get("keywords", [])
            for keyword in understanding["keywords"]:
                if keyword.lower() in [k.lower() for k in msg_keywords]:
                    score += 1.5
        
        # === CATEGORY MATCHING ===
        if category and understanding.get("category_hints"):
            for hint in understanding["category_hints"]:
                if hint.lower() in category.name.lower():
                    score += 4.0
        
        # === RECENCY BOOST ===
        age_days = (datetime.utcnow() - message.created_at).days
        if age_days < 1:
            score += 3.0
        elif age_days < 7:
            score += 2.0
        elif age_days < 30:
            score += 1.0
        
        # === INTENT MATCHING ===
        intent = understanding.get("intent", "")
        if message.tags:
            msg_intent = message.tags.get("actionables", [])
            if intent == "find_specific" and msg_intent:
                score += 2.0
        
        return score
    
    def _generate_preview(self, content: str, keywords: List[str]) -> str:
        """Generate a preview highlighting relevant parts"""
        if len(content) <= 150:
            return content
        
        # Find first keyword occurrence
        content_lower = content.lower()
        for keyword in keywords:
            idx = content_lower.find(keyword.lower())
            if idx != -1:
                # Get context around keyword
                start = max(0, idx - 50)
                end = min(len(content), idx + 100)
                preview = content[start:end]
                if start > 0:
                    preview = "..." + preview
                if end < len(content):
                    preview = preview + "..."
                return preview
        
        # No keyword found, return first 150 chars
        return content[:150] + "..."
    
    def _get_time_cutoff(self, time_filter: str) -> Optional[datetime]:
        """Get datetime cutoff for time filters"""
        now = datetime.utcnow()
        
        if time_filter == "today":
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif time_filter == "this_week":
            return now - timedelta(days=7)
        elif time_filter == "last_month":
            return now - timedelta(days=30)
        
        return None
    
    async def _get_user(self, phone: str, db: AsyncSession) -> Optional[User]:
        result = await db.execute(select(User).where(User.phone_number == phone))
        return result.scalar_one_or_none()
    
    async def _get_user_categories(self, user_id: int, db: AsyncSession) -> List[str]:
        result = await db.execute(
            select(Category.name).where(Category.user_id == user_id)
        )
        return [row[0] for row in result.all()]