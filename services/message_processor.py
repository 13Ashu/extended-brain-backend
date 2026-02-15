"""
Message Processor Service
Handles incoming messages, categorization, and storage

IMPORTANT: Save this file as services/message_processor.py
Create a 'services' folder and put this file inside it
"""

from typing import Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime

from database import User, Message, Category, MessageType
from cerebras_client import CerebrasClient


class MessageProcessor:
    """Process and store incoming messages"""
    
    def __init__(self, cerebras_client: CerebrasClient):
        self.cerebras = cerebras_client
    
    async def process(
        self,
        user_phone: str,
        content: str,
        message_type: str,
        media_url: Optional[str],
        db: AsyncSession
    ) -> Dict:
        """
        Process incoming message:
        1. Get or create user
        2. Analyze content with AI
        3. Categorize and tag
        4. Store in database
        
        Returns processing result
        """
        
        # Get or create user
        user = await self._get_or_create_user(user_phone, db)
        
        # Get existing categories for this user
        existing_categories = await self._get_user_categories(user.id, db)
        category_names = [cat.name for cat in existing_categories]
        
        # Analyze message with Cerebras
        analysis = await self.cerebras.categorize_message(
            content=content,
            existing_categories=category_names,
            message_type=message_type
        )
        
        # Get or create category
        category = await self._get_or_create_category(
            user_id=user.id,
            category_name=analysis["category"],
            db=db
        )
        
        # Create message record
        message = Message(
            user_id=user.id,
            category_id=category.id,
            content=content,
            message_type=MessageType[message_type.upper()],
            media_url=media_url,
            summary=analysis.get("summary"),
            tags=analysis.get("tags", []),
            entities=analysis.get("entities", {}),
            original_timestamp=datetime.utcnow()
        )
        
        # Generate embedding for semantic search (if needed)
        # embedding = await self.cerebras.generate_embedding(content)
        # message.embedding = str(embedding)
        
        db.add(message)
        await db.commit()
        await db.refresh(message)
        
        return {
            "message_id": message.id,
            "category": category.name,
            "tags": analysis.get("tags", []),
            "summary": analysis.get("summary"),
            "entities": analysis.get("entities", {})
        }
    
    async def _get_or_create_user(
        self,
        phone_number: str,
        db: AsyncSession
    ) -> User:
        """Get existing user or create new one"""
        
        result = await db.execute(
            select(User).where(User.phone_number == phone_number)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            user = User(phone_number=phone_number)
            db.add(user)
            await db.commit()
            await db.refresh(user)
        
        return user
    
    async def _get_user_categories(
        self,
        user_id: int,
        db: AsyncSession
    ) -> list:
        """Get all categories for a user"""
        
        result = await db.execute(
            select(Category).where(Category.user_id == user_id)
        )
        return result.scalars().all()
    
    async def _get_or_create_category(
        self,
        user_id: int,
        category_name: str,
        db: AsyncSession
    ) -> Category:
        """Get existing category or create new one"""
        
        result = await db.execute(
            select(Category).where(
                Category.user_id == user_id,
                Category.name == category_name
            )
        )
        category = result.scalar_one_or_none()
        
        if not category:
            category = Category(
                user_id=user_id,
                name=category_name
            )
            db.add(category)
            await db.commit()
            await db.refresh(category)
        
        return category
