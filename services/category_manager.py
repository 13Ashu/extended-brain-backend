"""
Category Manager Service
Handle creation, editing, deletion of categories

IMPORTANT: Save this file as services/category_manager.py
Create a 'services' folder and put this file inside it
"""

from typing import List, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from database import User, Category, Message
from cerebras_client import CerebrasClient


class CategoryManager:
    """Manage user categories"""
    
    def __init__(self, cerebras_client: CerebrasClient):
        self.cerebras = cerebras_client
    
    async def list_categories(
        self,
        user_phone: str,
        db: AsyncSession
    ) -> List[Dict]:
        """
        Get all categories for a user with message counts
        """
        
        user = await self._get_user(user_phone, db)
        if not user:
            return []
        
        # Get categories with message counts
        stmt = (
            select(
                Category,
                func.count(Message.id).label("message_count")
            )
            .outerjoin(Message, Message.category_id == Category.id)
            .where(Category.user_id == user.id)
            .group_by(Category.id)
            .order_by(Category.name)
        )
        
        result = await db.execute(stmt)
        rows = result.all()
        
        return [
            {
                "id": category.id,
                "name": category.name,
                "description": category.description,
                "color": category.color,
                "icon": category.icon,
                "count": count,
                "created_at": category.created_at.isoformat(),
                "updated_at": category.updated_at.isoformat()
            }
            for category, count in rows
        ]
    
    async def create_category(
        self,
        user_phone: str,
        name: str,
        db: AsyncSession,
        description: Optional[str] = None,
    ) -> Dict:
        """Create a new category"""
        
        user = await self._get_user(user_phone, db)
        if not user:
            raise ValueError("User not found")
        
        # Check if category already exists
        existing = await db.execute(
            select(Category).where(
                and_(
                    Category.user_id == user.id,
                    Category.name == name
                )
            )
        )
        
        if existing.scalar_one_or_none():
            raise ValueError(f"Category '{name}' already exists")
        
        # Create new category
        category = Category(
            user_id=user.id,
            name=name,
            description=description
        )
        
        db.add(category)
        await db.commit()
        await db.refresh(category)
        
        return {
            "id": category.id,
            "name": category.name,
            "description": category.description,
            "created_at": category.created_at.isoformat()
        }
    
    async def edit_category(
        self,
        user_phone: str,
        old_name: str,
        db: AsyncSession,
        new_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict:
        """Edit an existing category"""
        
        user = await self._get_user(user_phone, db)
        if not user:
            raise ValueError("User not found")
        
        # Get category
        result = await db.execute(
            select(Category).where(
                and_(
                    Category.user_id == user.id,
                    Category.name == old_name
                )
            )
        )
        category = result.scalar_one_or_none()
        
        if not category:
            raise ValueError(f"Category '{old_name}' not found")
        
        # Update fields
        if new_name:
            # Check if new name already exists
            existing = await db.execute(
                select(Category).where(
                    and_(
                        Category.user_id == user.id,
                        Category.name == new_name,
                        Category.id != category.id
                    )
                )
            )
            if existing.scalar_one_or_none():
                raise ValueError(f"Category '{new_name}' already exists")
            
            category.name = new_name
        
        if description is not None:
            category.description = description
        
        await db.commit()
        await db.refresh(category)
        
        return {
            "id": category.id,
            "name": category.name,
            "description": category.description,
            "updated_at": category.updated_at.isoformat()
        }
    
    async def delete_category(
        self,
        user_phone: str,
        name: str,
        db: AsyncSession,
        reassign_to: Optional[str] = "Uncategorized",
    ) -> Dict:
        """
        Delete a category
        """
        
        user = await self._get_user(user_phone, db)
        if not user:
            raise ValueError("User not found")
        
        # Get category to delete
        result = await db.execute(
            select(Category).where(
                and_(
                    Category.user_id == user.id,
                    Category.name == name
                )
            )
        )
        category = result.scalar_one_or_none()
        
        if not category:
            raise ValueError(f"Category '{name}' not found")
        
        # Get or create reassignment category
        reassign_category = None
        if reassign_to:
            result = await db.execute(
                select(Category).where(
                    and_(
                        Category.user_id == user.id,
                        Category.name == reassign_to
                    )
                )
            )
            reassign_category = result.scalar_one_or_none()
            
            if not reassign_category:
                reassign_category = Category(
                    user_id=user.id,
                    name=reassign_to
                )
                db.add(reassign_category)
                await db.flush()
        
        # Get messages in this category
        messages_result = await db.execute(
            select(Message).where(Message.category_id == category.id)
        )
        messages = messages_result.scalars().all()
        
        # Reassign messages
        for message in messages:
            message.category_id = reassign_category.id if reassign_category else None
        
        # Delete category
        await db.delete(category)
        await db.commit()
        
        return {
            "deleted": name,
            "messages_reassigned": len(messages),
            "reassigned_to": reassign_to
        }
    
    async def _get_user(self, phone_number: str, db: AsyncSession) -> Optional[User]:
        """Get user by phone number"""
        result = await db.execute(
            select(User).where(User.phone_number == phone_number)
        )
        return result.scalar_one_or_none()
