"""
List Service
─────────────────────────────────────────────────────────────
Owns all named-list logic: shopping list, bag list, packing list, etc.

A "list" is a single Message row where:
  - tags.list_type  = "shopping" | "bag" | "packing" | "grocery" | "custom"
  - tags.list_name  = "Shopping List" (normalized display name)
  - tags.subtasks   = [{task, done, added_at}, ...]
  - tags.all_buckets = ["List"]
  - tags.is_active  = True (False when archived)

Zero hallucination: list contents are NEVER passed through LLM for display.
All reads are direct DB fetches → render.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from sqlalchemy import and_, select, update, text, func
from sqlalchemy.ext.asyncio import AsyncSession

from database import async_session_maker, Message, User, Category
from cerebras_client import CerebrasClient


# ─────────────────────────────────────────────────────────────────────────────
# List type detection
# ─────────────────────────────────────────────────────────────────────────────

LIST_TYPE_MAP = {
    "shopping":  ["shopping list", "grocery list", "groceries", "supermarket list"],
    "bag":       ["bag list", "bag pack", "bag checklist", "school bag", "exam bag"],
    "packing":   ["packing list", "travel list", "trip list", "luggage list", "what to pack"],
    "todo":      ["todo list", "to-do list", "task list"],
    "reading":   ["reading list", "books to read", "book list"],
    "watching":  ["watch list", "movies to watch", "shows to watch"],
    "custom":    [],  # fallback
}

# Signals that this is an ADD operation on an existing list
ADD_SIGNALS = [
    r"add (?:to|in|into) [\w\s]+list",
    r"add (?:to|in|into) [\w\s]+bag",
    r"put (?:in|into) [\w\s]+list",
    r"[\w\s]+list[:\s]*[\n•\-\*]",
    r"add .+ to (my )?[\w\s]+ list",
]

# Signals that this is a SHOW operation
SHOW_SIGNALS = [
    r"show [\w\s]*list",
    r"[\w\s]*list\?$",
    r"what(?:'s| is) (?:in|on) (?:my )?[\w\s]*list",
    r"[\w\s]*list today",
    r"my [\w\s]*list",
    r"get [\w\s]*list",
]

# Signals that this is a CREATE/UPDATE operation
CREATE_SIGNALS = [
    r"[\w\s]+list:\s*\n",
    r"[\w\s]+list:\s*[\n•\-\*]",
    r"^[\w\s]*list[:\s]*$",
]


class ListService:

    def __init__(self, cerebras_client: CerebrasClient):
        self.cerebras = cerebras_client

    # ──────────────────────────────────────────────────────────────
    # Intent detection
    # ──────────────────────────────────────────────────────────────

    def detect_list_intent(self, content: str) -> Optional[Dict]:
        """
        Returns {intent, list_type, list_name, items} or None if not a list operation.
        intent: "create" | "add" | "show" | "clear_done"
        """
        lc = content.lower().strip()

        # SHOW intent
        for pattern in SHOW_SIGNALS:
            if re.search(pattern, lc):
                list_type, list_name = self._extract_list_type(lc)
                if list_type:
                    return {
                        "intent":    "show",
                        "list_type": list_type,
                        "list_name": list_name,
                        "items":     [],
                    }

        # ADD intent
        for pattern in ADD_SIGNALS:
            if re.search(pattern, lc):
                list_type, list_name = self._extract_list_type(lc)
                items = self._extract_items(content)
                if list_type and items:
                    return {
                        "intent":    "add",
                        "list_type": list_type,
                        "list_name": list_name,
                        "items":     items,
                    }

        # CREATE intent — "shopping list:\n- item1\n- item2"
        for pattern in CREATE_SIGNALS:
            if re.search(pattern, lc):
                list_type, list_name = self._extract_list_type(lc)
                items = self._extract_items(content)
                if list_type and items:
                    return {
                        "intent":    "create",
                        "list_type": list_type,
                        "list_name": list_name,
                        "items":     items,
                    }

        return None

    def _extract_list_type(self, lc: str) -> Tuple[Optional[str], str]:
        """Returns (list_type, display_name)"""
        for list_type, signals in LIST_TYPE_MAP.items():
            if list_type == "custom":
                continue
            for sig in signals:
                if sig in lc:
                    return list_type, self._normalize_name(sig)

        # More flexible: look for known type words anywhere in query
        type_words = {
            "shopping": "shopping", "grocery": "shopping", "groceries": "shopping",
            "bag": "bag", "packing": "packing", "travel": "packing",
            "reading": "reading", "watch": "watching",
        }
        for word, list_type in type_words.items():
            if re.search(rf"\b{word}\b", lc):
                display = self._normalize_name(f"{word} list")
                return list_type, display

        # Fallback: "X list" or "X bag" pattern
        m = re.search(r"([\w]+)\s+(?:list|bag|checklist)", lc)
        if m:
            raw = m.group(1).strip()
            if raw and len(raw) < 30 and raw not in {"my", "the", "a", "an", "your", "show", "get"}:
                return "custom", self._normalize_name(raw + " list")

        return None, ""

    def _normalize_name(self, raw: str) -> str:
        """'shopping list' → 'Shopping List'"""
        return " ".join(w.capitalize() for w in raw.strip().split())

    def _extract_items(self, content: str) -> List[str]:
        """Extract bullet/numbered items from content."""
        items = []

        # Bullet patterns: -, *, •, numbered
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            # Remove bullet markers
            m = re.match(r"^[-*•\d+\.]\s*(.+)$", line)
            if m:
                item = m.group(1).strip()
                if item and len(item) > 0:
                    items.append(item)

        # If no bullets found, try comma-separated on same line
        if not items:
            # Look for "add X, Y, Z to list"
            m = re.search(r"(?:add|put)\s+(.+?)\s+(?:to|in|into)", content.lower())
            if m:
                raw = m.group(1)
                items = [i.strip() for i in re.split(r"[,;]", raw) if i.strip()]

        return [i for i in items if len(i) > 0 and len(i) < 200]

    # ──────────────────────────────────────────────────────────────
    # Core operations
    # ──────────────────────────────────────────────────────────────

    async def get_active_list(
        self, user_id: int, list_type: str, db: AsyncSession
    ) -> Optional[Message]:
        """Find the active (non-archived) list of given type."""
        result = await db.execute(
            select(Message)
            .where(
                and_(
                    Message.user_id == user_id,
                    text("messages.tags->>'list_type' = :lt"),
                    text("(messages.tags->>'is_active')::boolean IS NOT FALSE"),
                    text("messages.tags->'all_buckets' @> '\"List\"'::jsonb"),
                )
            )
            .params(lt=list_type)
            .order_by(Message.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def create_list(
        self,
        user_id: int,
        list_type: str,
        list_name: str,
        items: List[str],
        db: AsyncSession,
    ) -> Message:
        """Create a new named list with initial items."""
        # Get or create List category
        cat = await db.scalar(
            select(Category).where(
                and_(Category.user_id == user_id, Category.name == "List")
            )
        )
        if not cat:
            cat = Category(user_id=user_id, name="List", description="Named lists")
            db.add(cat)
            await db.flush()

        now = datetime.utcnow()
        subtasks = [
            {"task": item, "done": False, "added_at": now.isoformat()}
            for item in items
        ]

        msg = Message(
            user_id=user_id,
            category_id=cat.id,
            content=list_name,
            message_type=__import__("models").MessageType("text"),
            summary=f"{list_name} — {len(items)} items",
            tags={
                "list_type":   list_type,
                "list_name":   list_name,
                "subtasks":    subtasks,
                "all_buckets": ["List"],
                "primary_bucket": "List",
                "is_active":   True,
                "item_count":  len(items),
            },
            created_at=now,
        )
        db.add(msg)
        await db.commit()
        await db.refresh(msg)
        return msg

    async def add_items(
        self,
        user_id: int,
        list_type: str,
        list_name: str,
        items: List[str],
        db: AsyncSession,
    ) -> Tuple[Message, int]:
        """
        Add items to existing list, or create if not found.
        Returns (message, added_count).
        """
        msg = await self.get_active_list(user_id, list_type, db)

        if not msg:
            # Auto-create the list
            msg = await self.create_list(user_id, list_type, list_name, items, db)
            return msg, len(items)

        tags = dict(msg.tags or {})
        existing = list(tags.get("subtasks", []))
        existing_lower = {s["task"].lower() for s in existing}

        added = 0
        now = datetime.utcnow().isoformat()
        for item in items:
            if item.lower() not in existing_lower:
                existing.append({"task": item, "done": False, "added_at": now})
                existing_lower.add(item.lower())
                added += 1

        tags["subtasks"]   = existing
        tags["item_count"] = len(existing)

        async with async_session_maker() as session:
            await session.execute(
                update(Message).where(Message.id == msg.id).values(tags=tags)
            )
            await session.commit()

        # Refresh
        async with async_session_maker() as session:
            msg = await session.scalar(select(Message).where(Message.id == msg.id))

        return msg, added

    async def complete_item(self, message_id: int, item_index: int) -> bool:
        """Mark a single list item as done."""
        async with async_session_maker() as session:
            msg = await session.scalar(
                select(Message).where(Message.id == message_id)
            )
            if not msg:
                return False

            tags     = dict(msg.tags or {})
            subtasks = list(tags.get("subtasks", []))

            if item_index >= len(subtasks):
                return False

            subtasks[item_index]["done"]    = True
            subtasks[item_index]["done_at"] = datetime.utcnow().isoformat()
            tags["subtasks"] = subtasks

            # If all done, mark list completed but keep active
            done_count = sum(1 for s in subtasks if s.get("done"))
            tags["done_count"] = done_count

            await session.execute(
                update(Message).where(Message.id == message_id).values(tags=tags)
            )
            await session.commit()
        return True

    async def clear_done_items(self, message_id: int) -> int:
        """Remove all done items from list. Returns count removed."""
        async with async_session_maker() as session:
            msg = await session.scalar(
                select(Message).where(Message.id == message_id)
            )
            if not msg:
                return 0

            tags     = dict(msg.tags or {})
            subtasks = list(tags.get("subtasks", []))
            before   = len(subtasks)
            subtasks = [s for s in subtasks if not s.get("done")]

            tags["subtasks"]   = subtasks
            tags["item_count"] = len(subtasks)

            await session.execute(
                update(Message).where(Message.id == message_id).values(tags=tags)
            )
            await session.commit()

        return before - len(subtasks)

    async def archive_list(self, message_id: int) -> bool:
        """Archive a list (hide from active)."""
        async with async_session_maker() as session:
            msg = await session.scalar(
                select(Message).where(Message.id == message_id)
            )
            if not msg:
                return False
            tags = dict(msg.tags or {})
            tags["is_active"] = False
            await session.execute(
                update(Message).where(Message.id == message_id).values(tags=tags)
            )
            await session.commit()
        return True

    async def get_all_user_lists(
        self, user_id: int, db: AsyncSession
    ) -> List[Message]:
        """Get all active lists for a user."""
        result = await db.execute(
            select(Message)
            .where(
                and_(
                    Message.user_id == user_id,
                    text("messages.tags->'all_buckets' @> '\"List\"'::jsonb"),
                    text("(messages.tags->>'is_active')::boolean IS NOT FALSE"),
                )
            )
            .order_by(Message.created_at.desc())
        )
        return result.scalars().all()

    # ──────────────────────────────────────────────────────────────
    # Display formatting — zero LLM
    # ──────────────────────────────────────────────────────────────

    def format_for_telegram(
        self, msg: Message
    ) -> Tuple[str, dict]:
        """
        Returns (text, reply_markup) for Telegram.
        Items rendered directly from DB — no LLM involved.
        """
        tags     = msg.tags if isinstance(msg.tags, dict) else {}
        subtasks = tags.get("subtasks", [])
        name     = tags.get("list_name", msg.content)

        pending = [s for s in subtasks if not s.get("done")]
        done    = [s for s in subtasks if s.get("done")]

        total    = len(subtasks)
        done_ct  = len(done)

        text = f"📋 *{name}*"
        if total > 0:
            text += f"  ·  _{done_ct}/{total} done_"
        text += "\n\n"

        if not subtasks:
            text += "_Nothing here yet._"
            return text, {"inline_keyboard": []}

        buttons = []

        # Pending items as tappable buttons
        for i, sub in enumerate(subtasks):
            if not sub.get("done"):
                global_idx = subtasks.index(sub)
                buttons.append([{
                    "text":          f"☐  {sub['task'][:45]}",
                    "callback_data": f"list_done:{msg.id}:{global_idx}",
                }])
            else:
                text += f"~{sub['task'][:50]}~\n"

        if done and pending:
            text += "\n"  # spacer between done/pending in text

        if done_ct == total and total > 0:
            text += "\n_All done! 🎉_\n"
            buttons.append([{
                "text":          "🗑 Clear done items",
                "callback_data": f"list_clear:{msg.id}",
            }])

        return text, {"inline_keyboard": buttons}

    def format_plain(self, msg: Message) -> str:
        """Plain text list for non-Telegram or fallback."""
        tags     = msg.tags if isinstance(msg.tags, dict) else {}
        subtasks = tags.get("subtasks", [])
        name     = tags.get("list_name", msg.content)

        lines = [f"📋 {name}\n"]
        for s in subtasks:
            prefix = "✓" if s.get("done") else "☐"
            lines.append(f"{prefix} {s['task']}")

        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────
    # Query helpers for search_service
    # ──────────────────────────────────────────────────────────────

    def is_list_query(self, query: str) -> bool:
        """Quick check if a search query is asking for a named list."""
        lc = query.lower()
        for list_type, signals in LIST_TYPE_MAP.items():
            if list_type == "custom":
                continue
            for sig in signals:
                if sig in lc:
                    return True
        return bool(re.search(r"[\w\s]+ list\??$", lc))

    def extract_list_type_from_query(self, query: str) -> Tuple[Optional[str], str]:
        return self._extract_list_type(query.lower())
