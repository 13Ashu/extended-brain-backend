"""
Subtask Service
─────────────────────────────────────────────────────────────
• Subtasks stored as JSONB array in parent message tags
• Supports command-style: "subtask: <parent> > <task1>, <task2>"
• Supports natural language with confirmation if ambiguous
• Complete individual subtasks via inline buttons
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import httpx
from sqlalchemy import and_, select, update, text, func
from sqlalchemy.ext.asyncio import AsyncSession

from database import async_session_maker, Message, User, Category
from cerebras_client import CerebrasClient


async def _send_telegram(chat_id: str, text_: str, reply_markup: Optional[dict] = None):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    payload = {"chat_id": chat_id, "text": text_, "parse_mode": "Markdown"}
    if reply_markup:
        payload["reply_markup"] = reply_markup
    async with httpx.AsyncClient(timeout=15.0) as client:
        await client.post(
            f"https://api.telegram.org/bot{token}/sendMessage", json=payload
        )


class SubtaskService:

    def __init__(self, cerebras_client: CerebrasClient):
        self.cerebras = cerebras_client

    # ──────────────────────────────────────────────────────────────
    # Parse subtask command
    # ──────────────────────────────────────────────────────────────

    def parse_command(self, content: str) -> Optional[Tuple[str, List[str]]]:
        """
        Parse: "subtask: <parent task> > <sub1>, <sub2>, <sub3>"
        Returns (parent_query, [subtasks]) or None if not a command.
        """
        lc = content.lower().strip()
        if not lc.startswith("subtask:"):
            return None

        body = content[len("subtask:"):].strip()
        if ">" not in body:
            return None

        parts   = body.split(">", 1)
        parent  = parts[0].strip()
        subs    = [s.strip() for s in parts[1].split(",") if s.strip()]

        if not parent or not subs:
            return None

        return parent, subs

    # ──────────────────────────────────────────────────────────────
    # Natural language detection
    # ──────────────────────────────────────────────────────────────

    async def detect_subtask_intent(
        self, content: str, user_id: int, db: AsyncSession
    ) -> Optional[Dict]:
        """
        Detect if a natural language message is trying to add subtasks.
        Returns {parent_query, subtasks} or None.
        """
        lc = content.lower()

        # Quick signal check before calling LLM
        subtask_signals = {
            "subtask", "sub-task", "break down", "breakdown",
            "split into", "steps for", "add to", "under", "part of"
        }
        if not any(sig in lc for sig in subtask_signals):
            return None

        # Get recent todos as candidates
        recent_todos = await self._get_recent_todos(user_id, db)
        todos_text = "\n".join(
            f"- ID {m.id}: {m.content[:80]}" for m in recent_todos[:10]
        )

        prompt = f"""Detect if the user is trying to add subtasks to an existing task.

USER MESSAGE: "{content}"

RECENT TODOS:
{todos_text if todos_text else "None"}

If the user is adding subtasks, return:
{{
  "is_subtask": true,
  "parent_id": <message_id or null if unclear>,
  "parent_hint": "the text they used to refer to the parent task",
  "subtasks": ["subtask 1", "subtask 2"],
  "confidence": 0.0
}}

If not a subtask request:
{{"is_subtask": false}}

Return ONLY JSON."""

        try:
            response = await self.cerebras.chat_lite(prompt, max_tokens=300)
            if response.get("is_subtask") and float(response.get("confidence", 0)) > 0.75:
                return response
        except Exception as e:
            print(f"[subtask] Detection failed: {e}")

        return None

    # ──────────────────────────────────────────────────────────────
    # Find parent message by query string
    # ──────────────────────────────────────────────────────────────

    async def find_parent_message(
        self, user_id: int, query: str, db: AsyncSession
    ) -> List[Message]:
        """Semantic search for parent task."""
        result = await db.execute(
            select(Message)
            .where(
                and_(
                    Message.user_id == user_id,
                    text("messages.tags->'all_buckets' @> '\"To-Do\"'::jsonb"),
                    text("(messages.tags->>'done')::boolean IS NOT TRUE"),
                    func.lower(Message.content).contains(query.lower()),
                )
            )
            .order_by(Message.created_at.desc())
            .limit(5)
        )
        return result.scalars().all()

    # ──────────────────────────────────────────────────────────────
    # Add subtasks to a message
    # ──────────────────────────────────────────────────────────────

    async def add_subtasks(
        self, message_id: int, subtasks: List[str]
    ) -> bool:
        try:
            async with async_session_maker() as session:
                msg = await session.scalar(
                    select(Message).where(Message.id == message_id)
                )
                if not msg:
                    return False

                tags = dict(msg.tags or {})
                existing = tags.get("subtasks", [])

                # Add new subtasks, avoiding duplicates
                existing_texts = {s["task"].lower() for s in existing}
                for sub in subtasks:
                    if sub.lower() not in existing_texts:
                        existing.append({"task": sub, "done": False})

                tags["subtasks"] = existing
                await session.execute(
                    update(Message)
                    .where(Message.id == message_id)
                    .values(tags=tags)
                )
                await session.commit()
            return True
        except Exception as e:
            print(f"[subtask] Add failed: {e}")
            return False

    # ──────────────────────────────────────────────────────────────
    # Complete a subtask by index
    # ──────────────────────────────────────────────────────────────

    async def complete_subtask(self, message_id: int, subtask_index: int) -> bool:
        try:
            async with async_session_maker() as session:
                msg = await session.scalar(
                    select(Message).where(Message.id == message_id)
                )
                if not msg:
                    return False

                tags     = dict(msg.tags or {})
                subtasks = tags.get("subtasks", [])

                if subtask_index >= len(subtasks):
                    return False

                subtasks[subtask_index]["done"] = True
                tags["subtasks"] = subtasks

                # If all subtasks done, mark parent done too
                if all(s["done"] for s in subtasks):
                    tags["done"]    = True
                    tags["done_at"] = __import__("datetime").datetime.utcnow().isoformat()

                await session.execute(
                    update(Message)
                    .where(Message.id == message_id)
                    .values(tags=tags)
                )
                await session.commit()
            return True
        except Exception as e:
            print(f"[subtask] Complete failed: {e}")
            return False

    # ──────────────────────────────────────────────────────────────
    # Format subtask view for Telegram
    # ──────────────────────────────────────────────────────────────

    def format_subtasks(self, message: Message) -> Tuple[str, dict]:
        """Returns (text, reply_markup) showing subtasks as inline buttons."""
        tags     = message.tags if isinstance(message.tags, dict) else {}
        subtasks = tags.get("subtasks", [])

        text_ = f"📋 *{message.content[:60]}*\n\n"
        buttons = []

        if not subtasks:
            text_ += "_No subtasks yet._"
            return text_, {"inline_keyboard": buttons}

        done_count = sum(1 for s in subtasks if s.get("done"))
        text_ += f"_{done_count}/{len(subtasks)} completed_\n\n"

        for i, sub in enumerate(subtasks):
            is_done = sub.get("done", False)
            if is_done:
                text_ += f"~✓ {sub['task'][:50]}~\n"
            else:
                text_ += f"• {sub['task'][:50]}\n"
                buttons.append([{
                    "text":          f"✓ {sub['task'][:35]}",
                    "callback_data": f"subtask:{message.id}:{i}",
                }])

        if done_count == len(subtasks):
            text_ += "\n_All subtasks complete! 🎉_"

        return text_, {"inline_keyboard": buttons}

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    async def _get_recent_todos(
        self, user_id: int, db: AsyncSession
    ) -> List[Message]:
        result = await db.execute(
            select(Message)
            .where(
                and_(
                    Message.user_id == user_id,
                    text("messages.tags->'all_buckets' @> '\"To-Do\"'::jsonb"),
                    text("(messages.tags->>'done')::boolean IS NOT TRUE"),
                )
            )
            .order_by(Message.created_at.desc())
            .limit(15)
        )
        return result.scalars().all()
