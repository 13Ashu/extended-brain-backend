"""
Search Service
Semantic and keyword-based search through user's knowledge base

IMPORTANT: Save this file as services/search_service.py
Create a 'services' folder and put this file inside it
"""

from typing import List, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, and_, func
from datetime import datetime, timedelta

from database import User, Message, Category
from cerebras_client import CerebrasClient


class SearchService:
    """Search through user's messages"""
    
    def __init__(self, cerebras_client: CerebrasClient):
        self.cerebras = cerebras_client
    
    async def search(
        self,
        user_phone: str,
        query: str,
        db: AsyncSession,
        limit: int = 10,
        category_filter: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Search user's knowledge base
        
        Search methods:
        1. Keyword matching in content, tags, summary
        2. Category filtering
        3. Semantic search (if embeddings available)
        4. Date-based filtering
        
        Returns list of matching messages
        """
        
        # Get user
        user = await self._get_user(user_phone, db)
        if not user:
            return []
        
        # Enhance query with AI
        enhanced = await self.cerebras.search_query_enhancement(query)
        search_terms = [query] + enhanced.get("synonyms", [])
        
        # Build base query
        stmt = (
            select(Message, Category)
            .join(Category, Message.category_id == Category.id, isouter=True)
            .where(Message.user_id == user.id)
        )
        
        # Apply text search across multiple fields
        text_conditions = []
        for term in search_terms:
            term_lower = f"%{term.lower()}%"
            text_conditions.append(
                or_(
                    func.lower(Message.content).like(term_lower),
                    func.lower(Message.summary).like(term_lower),
                    func.lower(Category.name).like(term_lower)
                )
            )
        
        if text_conditions:
            stmt = stmt.where(or_(*text_conditions))
        
        # Apply category filter
        if category_filter:
            stmt = stmt.where(Category.name.in_(category_filter))
        
        # Order by relevance (most recent first for now)
        stmt = stmt.order_by(Message.created_at.desc()).limit(limit)
        
        # Execute query
        result = await db.execute(stmt)
        rows = result.all()
        
        # Format results
        results = []
        for message, category in rows:
            results.append({
                "id": message.id,
                "content": message.content,
                "summary": message.summary,
                "category": category.name if category else "Uncategorized",
                "tags": message.tags or [],
                "message_type": message.message_type.value,
                "media_url": message.media_url,
                "created_at": message.created_at.isoformat(),
                "relevance_score": self._calculate_relevance(message, query)
            })
        
        # Sort by relevance
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return results
    
    async def search_by_date(
        self,
        user_phone: str,
        db: AsyncSession,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict]:
        """Search messages by date range"""
        
        user = await self._get_user(user_phone, db)
        if not user:
            return []
        
        stmt = (
            select(Message, Category)
            .join(Category, Message.category_id == Category.id, isouter=True)
            .where(Message.user_id == user.id)
        )
        
        if start_date:
            stmt = stmt.where(Message.created_at >= start_date)
        
        if end_date:
            stmt = stmt.where(Message.created_at <= end_date)
        
        stmt = stmt.order_by(Message.created_at.desc())
        
        result = await db.execute(stmt)
        rows = result.all()
        
        return [
            {
                "id": msg.id,
                "content": msg.content,
                "category": cat.name if cat else "Uncategorized",
                "created_at": msg.created_at.isoformat()
            }
            for msg, cat in rows
        ]
    
    async def search_by_category(
        self,
        user_phone: str,
        category_name: str,
        db: AsyncSession,
        limit: int = 50,
    ) -> List[Dict]:
        """Get all messages in a specific category"""
        
        user = await self._get_user(user_phone, db)
        if not user:
            return []
        
        stmt = (
            select(Message)
            .join(Category)
            .where(
                and_(
                    Message.user_id == user.id,
                    Category.name == category_name
                )
            )
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        
        result = await db.execute(stmt)
        messages = result.scalars().all()
        
        return [
            {
                "id": msg.id,
                "content": msg.content,
                "summary": msg.summary,
                "tags": msg.tags,
                "created_at": msg.created_at.isoformat()
            }
            for msg in messages
        ]
    
    async def get_recent_messages(
        self,
        user_phone: str,
        db: AsyncSession,
        days: int = 7,
        limit: int = 20,
    ) -> List[Dict]:
        """Get recent messages from last N days"""
        
        since = datetime.utcnow() - timedelta(days=days)
        return await self.search_by_date(
            user_phone=user_phone,
            start_date=since,
            db=db
        )
    
    def _calculate_relevance(self, message: Message, query: str) -> float:
        """
        Calculate relevance score for a message
        Simple scoring for now - can be enhanced with:
        - TF-IDF
        - Embedding similarity
        - Recency boost
        """
        score = 0.0
        query_lower = query.lower()
        
        # Content match
        if query_lower in message.content.lower():
            score += 3.0
        
        # Summary match
        if message.summary and query_lower in message.summary.lower():
            score += 2.0
        
        # Tag match
        if message.tags:
            for tag in message.tags:
                if query_lower in tag.lower():
                    score += 1.5
        
        # Recency boost (messages from last week get bonus)
        age_days = (datetime.utcnow() - message.created_at).days
        if age_days < 7:
            score += (7 - age_days) * 0.1
        
        return score
    
    async def _get_user(self, phone_number: str, db: AsyncSession) -> Optional[User]:
        """Get user by phone number"""
        result = await db.execute(
            select(User).where(User.phone_number == phone_number)
        )
        return result.scalar_one_or_none()
