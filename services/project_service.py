"""
Project Service
─────────────────────────────────────────────────────────────
• Auto-detects project from message concepts/keywords
• Always confirms with user before grouping
• Supports both command-style and natural language
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import httpx
from sqlalchemy import and_, select, update, text
from sqlalchemy.ext.asyncio import AsyncSession

from database import async_session_maker, Message, User
from cerebras_client import CerebrasClient


async def _send_telegram(chat_id: str, text_: str, reply_markup: Optional[dict] = None):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    payload = {
        "chat_id":    chat_id,
        "text":       text_,
        "parse_mode": "Markdown",
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup
    async with httpx.AsyncClient(timeout=15.0) as client:
        await client.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json=payload,
        )


class ProjectService:

    def __init__(self, cerebras_client: CerebrasClient):
        self.cerebras = cerebras_client

    # ──────────────────────────────────────────────────────────────
    # Called after every save — detect if message belongs to a project
    # ──────────────────────────────────────────────────────────────

    async def detect_and_suggest(
        self,
        user: User,
        message: Message,
        analysis: Dict,
        db: AsyncSession,
    ) -> Optional[str]:
        """
        Returns a suggested project name if detected, else None.
        Caller should store pending confirmation in context_service
        and send the confirmation message to user.
        """
        # Get existing projects for this user
        existing = await self._get_user_projects(user.id, db)

        concepts = analysis.get("concepts", [])
        keywords = analysis.get("keywords", [])
        content  = message.content

        prompt = f"""You are detecting if a personal note belongs to an ongoing project.

USER'S EXISTING PROJECTS: {existing if existing else "None yet"}

NOTE: "{content}"
CONCEPTS: {concepts}
KEYWORDS: {keywords}

TASK:
1. If this note clearly belongs to one of the existing projects, return that project name.
2. If this note suggests a NEW project (recurring theme, multi-step effort), suggest a short project name.
3. If this is just a standalone note with no project context, return null.

RULES:
- Only suggest a project if you are >80% confident
- Project names should be short: "Japan Trip", "Home Renovation", "Work Q2"
- Do NOT suggest projects for single todos like "buy milk"

Return ONLY this JSON:
{{
  "project": "Project Name or null",
  "confidence": 0.0,
  "is_new": true,
  "reason": "one line explanation"
}}"""

        try:
            response = await self.cerebras.chat_lite(prompt, max_tokens=200)
            project  = response.get("project")
            confidence = float(response.get("confidence", 0))

            if project and confidence >= 0.80:
                return project
        except Exception as e:
            print(f"[project] Detection failed: {e}")

        return None

    # ──────────────────────────────────────────────────────────────
    # Confirm and assign project to a message
    # ──────────────────────────────────────────────────────────────

    async def assign_project(self, message_id: int, project_name: str) -> bool:
        try:
            async with async_session_maker() as session:
                msg = await session.scalar(
                    select(Message).where(Message.id == message_id)
                )
                if not msg:
                    return False
                tags = dict(msg.tags or {})
                tags["project"] = project_name
                await session.execute(
                    update(Message)
                    .where(Message.id == message_id)
                    .values(tags=tags)
                )
                await session.commit()
            return True
        except Exception as e:
            print(f"[project] Assign failed: {e}")
            return False

    # ──────────────────────────────────────────────────────────────
    # List all messages in a project
    # ──────────────────────────────────────────────────────────────

    async def get_project_messages(
        self, user_id: int, project_name: str, db: AsyncSession
    ) -> List[Message]:
        result = await db.execute(
            select(Message)
            .where(
                and_(
                    Message.user_id == user_id,
                    text("messages.tags->>'project' ILIKE :p"),
                )
            )
            .params(p=f"%{project_name}%")
            .order_by(Message.created_at.desc())
        )
        return result.scalars().all()

    # ──────────────────────────────────────────────────────────────
    # List all projects for a user
    # ──────────────────────────────────────────────────────────────

    async def _get_user_projects(self, user_id: int, db: AsyncSession) -> List[str]:
        result = await db.execute(
            select(text("DISTINCT messages.tags->>'project'"))
            .where(
                and_(
                    Message.user_id == user_id,
                    text("messages.tags->>'project' IS NOT NULL"),
                )
            )
        )
        return [row[0] for row in result.all() if row[0]]

    async def get_user_projects_with_counts(
        self, user_id: int, db: AsyncSession
    ) -> List[Dict]:
        result = await db.execute(
            select(
                text("messages.tags->>'project' AS project"),
                text("COUNT(*) AS count"),
            )
            .where(
                and_(
                    Message.user_id == user_id,
                    text("messages.tags->>'project' IS NOT NULL"),
                )
            )
            .group_by(text("messages.tags->>'project'"))
            .order_by(text("count DESC"))
        )
        return [{"name": row[0], "count": row[1]} for row in result.all()]

    # ──────────────────────────────────────────────────────────────
    # Format project summary for Telegram
    # ──────────────────────────────────────────────────────────────

    def format_project_summary(
        self, project_name: str, messages: List[Message]
    ) -> str:
        if not messages:
            return f"No items found in project *{project_name}*."

        text_ = f"📁 *Project: {project_name}*\n_{len(messages)} items_\n\n"

        todos  = [m for m in messages if "To-Do" in (m.tags or {}).get("all_buckets", [])]
        ideas  = [m for m in messages if "Ideas" in (m.tags or {}).get("all_buckets", [])]
        others = [m for m in messages if m not in todos and m not in ideas]

        if todos:
            text_ += "✅ *Tasks*\n"
            for m in todos[:5]:
                tags   = m.tags or {}
                done   = "~" if tags.get("done") else "•"
                text_ += f"{done} {m.content[:60]}\n"
            text_ += "\n"

        if ideas:
            text_ += "💡 *Ideas*\n"
            for m in ideas[:3]:
                text_ += f"• {m.content[:60]}\n"
            text_ += "\n"

        if others:
            text_ += "📝 *Notes*\n"
            for m in others[:3]:
                text_ += f"• {m.content[:60]}\n"

        return text_
