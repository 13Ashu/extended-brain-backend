"""
List Service — v2
─────────────────────────────────────────────────────────────
Named lists: each list has a unique name per user.
"Mall Shopping List" and "Dmart Shopping List" are separate lists.

Identity: (user_id, normalized_list_name) — NOT (user_id, list_type)

Zero hallucination: list contents NEVER go through LLM for display.
All reads are direct DB fetches → render.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from sqlalchemy import and_, or_, select, update, text
from sqlalchemy.ext.asyncio import AsyncSession

from database import async_session_maker, Message, User, Category
from cerebras_client import CerebrasClient


# ─────────────────────────────────────────────────────────────────────────────
# List type classification (for grouping, not identity)
# ─────────────────────────────────────────────────────────────────────────────

LIST_TYPE_SIGNALS = {
    "shopping": ["shopping", "grocery", "groceries", "supermarket", "market", "store", "dmart", "mall"],
    "bag":      ["bag", "school bag", "exam bag", "backpack"],
    "packing":  ["packing", "travel", "trip", "luggage", "suitcase", "vacation"],
    "reading":  ["reading", "books", "book list"],
    "watching": ["watch", "movies", "shows", "series"],
    "todo":     ["todo", "to-do", "task list"],
}

SKIP_WORDS = {
    "my", "the", "a", "an", "your", "show", "get", "give",
    "what", "whats", "please", "can", "you", "is", "are",
    "for", "of", "in", "on", "at", "to", "from",
}

# Patterns that clearly indicate list CREATE/ADD intent
# Must have a name AND bullet items
LIST_HEADER_PATTERNS = [
    # "Shopping list for mall:\n- item"  or  "dmart shopping:\n- item"
    r"^([\w][\w\s\-]+?):\s*\n\s*[-*•]",
    # "add to/in X list: ..."
    r"add (?:to|in|into) ([\w\s]+(?:list|bag))[\s:,]",
]

# Patterns that indicate SHOW intent
LIST_SHOW_PATTERNS = [
    # "show my dmart list today" / "get shopping list" / "give mall list"
    r"(?:show|get|give|display)\s+(?:my\s+|the\s+)?(\w[\w\s\-]+?(?:list|bag))(?:\s+today|\s+now)?\s*\??$",
    # "my dmart shopping list?" / "my bag list today?"
    r"^(?:my\s+)?(\w[\w\s\-]+?(?:list|bag))(?:\s+today|\s+now)?\s*\?+\s*$",
    # "whats in my shopping list" / "what is in my bag list"
    r"what(?:\'s| is) (?:in|on) (?:my\s+|the\s+)?([\w][\w\s\-]+?(?:list|bag))",
    # bare "dmart list?" or "shopping?"
    r"^(\w[\w\s\-]+?(?:list|bag))(?:\s+today|\s+now)?\s*\?+\s*$",
]

def _classify_list_type(name: str) -> str:
    lc = name.lower()
    for list_type, signals in LIST_TYPE_SIGNALS.items():
        if any(sig in lc for sig in signals):
            return list_type
    return "custom"


def _normalize_list_name(raw: str) -> str:
    """
    'dmart shopping' → 'Dmart Shopping List'
    'mall shopping list' → 'Mall Shopping List'
    'shopping list for mall' → 'Mall Shopping List' (reorders context)
    """
    raw = raw.strip().lower()
    raw = re.sub(r"[:\-\.\,]+$", "", raw).strip()
    raw = re.sub(r"^(my|the|a|an)\s+", "", raw)
    # Simplify "X list for Y" → "Y X List"
    m = re.match(r"(.+?)\s+(?:list|bag|checklist)\s+for\s+(.+)", raw)
    if m:
        base    = m.group(1).strip()
        context = m.group(2).strip()
        raw     = f"{context} {base} list"
    # Add "List" if missing
    if not re.search(r"\b(list|bag|checklist)\b", raw):
        raw = raw + " list"
    return " ".join(w.capitalize() for w in raw.split())


def _extract_items_from_content(content: str) -> List[str]:
    """
    Extract bullet/numbered items. Skips the header line.
    Handles: -, *, •, numbers.
    """
    items   = []
    lines   = content.split("\n")
    # Skip the first non-empty line (it's the header/name)
    skipped = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not skipped:
            skipped = True
            continue  # skip header line

        # Match bullet markers
        m = re.match(r"^[-*•]\s*(.+)$", line)
        if m:
            item = m.group(1).strip()
            # Split "item1, - item2" patterns (multiple items on one bullet line)
            sub_items = re.split(r",\s*[-*•]\s*", item)
            if len(sub_items) > 1:
                items.extend([s.strip() for s in sub_items if s.strip()])
            else:
                items.append(item)
            continue

        # Numbered: "1. item" or "1) item"
        m = re.match(r"^\d+[.)]\s*(.+)$", line)
        if m:
            items.append(m.group(1).strip())
            continue

        # Bare line that looks like an item (short, no colon at end)
        if line and not line.endswith(":") and len(line.split()) <= 8:
            items.append(line)

    # Fallback: comma-separated inline "add pen, mug, glue to list"
    if not items:
        m = re.search(r"(?:add|put)\s+(.+?)\s+(?:to|in|into)", content.lower())
        if m:
            items = [i.strip() for i in re.split(r"[,;]", m.group(1)) if i.strip()]

    return [i for i in items if 0 < len(i) < 200]


class ListService:

    def __init__(self, cerebras_client: CerebrasClient):
        self.cerebras = cerebras_client

    # ──────────────────────────────────────────────────────────────
    # Intent detection
    # ──────────────────────────────────────────────────────────────

    async def detect_list_intent(self, content: str) -> Optional[Dict]:
        """
        Detect list intent using LLM as primary, regex as fallback.
        Returns {intent, list_name, list_type, items} or None.
        intent: "create_or_add" | "show"
        """
        lc = content.lower().strip()

        # Fast pre-filter: skip obvious non-list messages before hitting LLM
        if lc.startswith(("search:", "find:", "/", "remind", "briefing:")):
            return None
        # CRITICAL: never intercept todo/task queries as list operations
        # "New todo list for tomorrow" must go to message_processor, not list_service
        TODO_BLOCK = {"todo", "to-do", "to do", "task", "tasks", "pending"}
        if any(kw in lc for kw in TODO_BLOCK):
            return None

        list_signals = [
            "list", "bag", "packing", "shopping", "grocery",
            "groceries", "checklist", "add to", "add in",
        ]
        if not any(sig in lc for sig in list_signals):
            return None

        # ── Primary: LLM ──────────────────────────────────────────
        try:
            result = await self._llm_detect(content)
            if result is not None:
                return result
        except Exception as e:
            print(f"[list] LLM detection failed, falling back to regex: {e}")

        # ── Fallback: regex ────────────────────────────────────────
        return self._regex_detect(content)

    async def _llm_detect(self, content: str) -> Optional[Dict]:
        """LLM-based list intent detection with strict JSON output."""
        prompt = (
            "You are classifying a user message for a personal knowledge base.\n\n"
            f"USER MESSAGE:\n{content}\n\n"
            "TASK: Classify into one of three categories:\n"
            "1. create_or_add — user is creating a named list or adding items to one\n"
            "2. show — user wants to see/retrieve a named list\n"
            "3. none — regular note, todo, reminder, search, anything else\n\n"
            "RULES:\n"
            "- A named list has a NAME and ITEMS under it (bullets or numbered)\n"
            "- The list_name MUST preserve the context word: dmart→Dmart, mall→Mall, japan→Japan\n"
            "- For show intent: infer full list name from query (dmart list → Dmart Shopping List)\n"
            "- todo batch with action items is NOT a list → none\n"
            "- reminders, notes, searches are NOT lists → none\n"
            "- Extract ALL bullet/numbered items, never include the header line in items\n\n"
            "EXAMPLES (study these carefully):\n\n"
            "Input: dmart shopping:\n- Pen\n- coffee mug\n- glue\n"
            'Output: {"intent":"create_or_add","list_name":"Dmart Shopping List","list_type":"shopping","items":["Pen","coffee mug","glue"]}\n\n'
            "Input: My shopping list for mall:\n- party shirt\n- brown belt\n"
            'Output: {"intent":"create_or_add","list_name":"Mall Shopping List","list_type":"shopping","items":["party shirt","brown belt"]}\n\n'
            "Input: Bag list for exam:\n- pencil\n- eraser\n- graph book\n"
            'Output: {"intent":"create_or_add","list_name":"Exam Bag List","list_type":"bag","items":["pencil","eraser","graph book"]}\n\n'
            "Input: add to dmart list: eggs, milk, bread\n"
            'Output: {"intent":"create_or_add","list_name":"Dmart Shopping List","list_type":"shopping","items":["eggs","milk","bread"]}\n\n'
            "Input: Show dmart list?\n"
            'Output: {"intent":"show","list_name":"Dmart Shopping List","list_type":"shopping","items":[]}\n\n'
            "Input: show my mall shopping list\n"
            'Output: {"intent":"show","list_name":"Mall Shopping List","list_type":"shopping","items":[]}\n\n'
            "Input: what's in my bag list?\n"
            'Output: {"intent":"show","list_name":"Bag List","list_type":"bag","items":[]}\n\n'
            "Input: get my exam bag list\n"
            'Output: {"intent":"show","list_name":"Exam Bag List","list_type":"bag","items":[]}\n\n'
            "Input: todo for today:\n- call mom\n- buy milk\n"
            'Output: {"intent":"none","list_name":null,"list_type":null,"items":[]}\n\n'
            "Input: remind me to drink water every 2 hours\n"
            'Output: {"intent":"none","list_name":null,"list_type":null,"items":[]}\n\n'
            "Return ONLY valid JSON, no markdown:\n"
            '{"intent":"...","list_name":"..."|null,"list_type":"shopping"|"bag"|"packing"|"reading"|"watching"|"custom"|null,"items":[]}'
        )
        response = await self.cerebras.chat(prompt, max_tokens=400)

        intent    = response.get("intent", "none")
        list_name = response.get("list_name")
        list_type = response.get("list_type")
        items     = response.get("items", [])

        if intent not in ("create_or_add", "show", "none"):
            return None
        if intent == "none" or not list_name:
            return None
        if intent == "create_or_add" and not items:
            return None
        if list_type not in ("shopping", "bag", "packing", "reading", "watching", "custom", None):
            list_type = "custom"

        return {
            "intent":    intent,
            "list_name": list_name,
            "list_type": list_type or "custom",
            "items":     [str(i).strip() for i in items if str(i).strip()],
        }

    def _regex_detect(self, content: str) -> Optional[Dict]:
        """Regex fallback — used when LLM fails or for sync is_list_query."""
        lc = content.lower().strip()

        for pattern in LIST_HEADER_PATTERNS:
            m = re.search(pattern, lc, re.MULTILINE)
            if m:
                raw_name = m.group(1).strip()
                if any(kw in raw_name for kw in {"todo", "to-do", "task", "remind"}):
                    continue
                items = _extract_items_from_content(content)
                if items:
                    list_name = _normalize_list_name(raw_name)
                    return {
                        "intent":    "create_or_add",
                        "list_name": list_name,
                        "list_type": _classify_list_type(list_name),
                        "items":     items,
                    }

        for pattern in LIST_SHOW_PATTERNS:
            m = re.search(pattern, lc)
            if m:
                raw_name = m.group(1).strip()
                if any(kw in raw_name for kw in {"todo", "to-do", "task", "pending"}):
                    continue
                meaningful = [w for w in raw_name.split() if w not in SKIP_WORDS and len(w) > 1 and w != "list"]
                if not meaningful:
                    continue
                list_name = _normalize_list_name(raw_name)
                return {
                    "intent":    "show",
                    "list_name": list_name,
                    "list_type": _classify_list_type(list_name),
                    "items":     [],
                }

        return None

    def is_list_query(self, query: str) -> bool:
        """Sync check using regex only — for search_service pre-filter."""
        result = self._regex_detect(query)
        return result is not None and result["intent"] == "show"


    # ──────────────────────────────────────────────────────────────
    # DB operations
    # ──────────────────────────────────────────────────────────────

    async def get_list_by_name(
        self, user_id: int, list_name: str, db: AsyncSession
    ) -> Optional[Message]:
        """Exact name match (case-insensitive)."""
        result = await db.execute(
            select(Message)
            .where(
                and_(
                    Message.user_id == user_id,
                    text("messages.tags->>'list_name' ILIKE :name"),
                    text("messages.tags->'all_buckets' @> '\"List\"'::jsonb"),
                    text("(messages.tags->>'is_active')::boolean IS NOT FALSE"),
                )
            )
            .params(name=list_name)
            .order_by(Message.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def find_best_matching_list(
        self, user_id: int, list_name: str, db: AsyncSession
    ) -> Optional[Message]:
        """
        Find best matching list — exact first, then fuzzy by key words.
        "Dmart Shopping List" matches "dmart list" or "dmart shopping".
        """
        # Exact match
        msg = await self.get_list_by_name(user_id, list_name, db)
        if msg:
            return msg

        # Key words from the query (excluding generic words)
        key_words = [
            w for w in list_name.lower().split()
            if w not in SKIP_WORDS | {"list", "bag", "checklist", "shopping", "grocery"}
            and len(w) > 2
        ]

        if not key_words:
            # No distinctive words — try type-based match
            list_type = _classify_list_type(list_name)
            if list_type != "custom":
                result = await db.execute(
                    select(Message)
                    .where(
                        and_(
                            Message.user_id == user_id,
                            text("messages.tags->>'list_type' = :lt"),
                            text("messages.tags->'all_buckets' @> '\"List\"'::jsonb"),
                            text("(messages.tags->>'is_active')::boolean IS NOT FALSE"),
                        )
                    )
                    .params(lt=list_type)
                    .order_by(Message.created_at.desc())
                    .limit(1)
                )
                return result.scalar_one_or_none()
            return None

        # Fuzzy: any key word appears in stored list name
        conditions = [
            text(f"messages.tags->>'list_name' ILIKE :kw{i}")
            for i, _ in enumerate(key_words[:4])
        ]
        params = {f"kw{i}": f"%{kw}%" for i, kw in enumerate(key_words[:4])}

        result = await db.execute(
            select(Message)
            .where(
                and_(
                    Message.user_id == user_id,
                    text("messages.tags->'all_buckets' @> '\"List\"'::jsonb"),
                    text("(messages.tags->>'is_active')::boolean IS NOT FALSE"),
                    or_(*conditions),
                )
            )
            .params(**params)
            .order_by(Message.created_at.desc())
            .limit(5)
        )
        candidates = result.scalars().all()

        if not candidates:
            return None

        # Score: count how many key words match
        def score(m: Message) -> int:
            stored = (m.tags or {}).get("list_name", "").lower()
            return sum(1 for kw in key_words if kw in stored)

        candidates.sort(key=score, reverse=True)
        # Return best scoring match, or most recent if no distinctive keywords matched
        return candidates[0]

    async def create_or_add(
        self,
        user_id: int,
        list_name: str,
        list_type: str,
        items: List[str],
        db: AsyncSession,
    ) -> Tuple[Message, int, bool]:
        """
        Create if not exists, add items if exists.
        Returns (message, items_added, was_created).
        """
        msg = await self.find_best_matching_list(user_id, list_name, db)

        if not msg:
            msg = await self._create_list(user_id, list_name, list_type, items, db)
            return msg, len(items), True

        # Add to existing
        tags           = dict(msg.tags or {})
        existing       = list(tags.get("subtasks", []))
        existing_lower = {s["task"].lower() for s in existing}
        added          = 0
        now            = datetime.utcnow().isoformat()

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
            msg = await session.scalar(select(Message).where(Message.id == msg.id))

        return msg, added, False

    async def _create_list(
        self, user_id: int, list_name: str, list_type: str,
        items: List[str], db: AsyncSession,
    ) -> Message:
        cat = await db.scalar(
            select(Category).where(
                and_(Category.user_id == user_id, Category.name == "List")
            )
        )
        if not cat:
            cat = Category(user_id=user_id, name="List", description="Named lists")
            db.add(cat)
            await db.flush()

        now      = datetime.utcnow()
        subtasks = [{"task": item, "done": False, "added_at": now.isoformat()} for item in items]

        msg = Message(
            user_id=user_id,
            category_id=cat.id,
            content=list_name,
            message_type=__import__("models").MessageType("text"),
            summary=f"{list_name} — {len(items)} items",
            tags={
                "list_type":      list_type,
                "list_name":      list_name,
                "subtasks":       subtasks,
                "all_buckets":    ["List"],
                "primary_bucket": "List",
                "is_active":      True,
                "item_count":     len(items),
            },
            created_at=now,
        )
        db.add(msg)
        await db.commit()
        await db.refresh(msg)
        return msg

    async def complete_item(self, message_id: int, item_index: int) -> bool:
        async with async_session_maker() as session:
            msg = await session.scalar(select(Message).where(Message.id == message_id))
            if not msg:
                return False
            tags     = dict(msg.tags or {})
            subtasks = list(tags.get("subtasks", []))
            if item_index >= len(subtasks):
                return False
            subtasks[item_index]["done"]    = True
            subtasks[item_index]["done_at"] = datetime.utcnow().isoformat()
            tags["subtasks"]   = subtasks
            tags["done_count"] = sum(1 for s in subtasks if s.get("done"))
            await session.execute(
                update(Message).where(Message.id == message_id).values(tags=tags)
            )
            await session.commit()
        return True

    async def clear_done_items(self, message_id: int) -> int:
        async with async_session_maker() as session:
            msg = await session.scalar(select(Message).where(Message.id == message_id))
            if not msg:
                return 0
            tags     = dict(msg.tags or {})
            subtasks = [s for s in tags.get("subtasks", []) if not s.get("done")]
            before   = tags.get("item_count", len(tags.get("subtasks", [])))
            tags["subtasks"]   = subtasks
            tags["item_count"] = len(subtasks)
            await session.execute(
                update(Message).where(Message.id == message_id).values(tags=tags)
            )
            await session.commit()
        return before - len(subtasks)

    async def get_all_user_lists(self, user_id: int, db: AsyncSession) -> List[Message]:
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
    # Display — zero LLM
    # ──────────────────────────────────────────────────────────────

    def format_for_telegram(self, msg: Message) -> Tuple[str, dict]:
        tags     = msg.tags if isinstance(msg.tags, dict) else {}
        subtasks = tags.get("subtasks", [])
        name     = tags.get("list_name", msg.content)
        done_ct  = sum(1 for s in subtasks if s.get("done"))
        total    = len(subtasks)

        text_ = f"📋 *{name}*"
        if total > 0:
            text_ += f"  ·  _{done_ct}/{total} done_"
        text_ += "\n\n"

        if not subtasks:
            text_ += "_Nothing here yet._"
            return text_, {"inline_keyboard": []}

        buttons = []
        for i, sub in enumerate(subtasks):
            if sub.get("done"):
                text_ += f"~{sub['task'][:50]}~\n"
            else:
                buttons.append([{
                    "text":          f"☐  {sub['task'][:45]}",
                    "callback_data": f"list_done:{msg.id}:{i}",
                }])

        if done_ct == total and total > 0:
            text_ += "\n_All done! 🎉_\n"
            buttons.append([{
                "text":          "🗑 Clear done items",
                "callback_data": f"list_clear:{msg.id}",
            }])

        return text_, {"inline_keyboard": buttons}

    def format_plain(self, msg: Message) -> str:
        tags     = msg.tags if isinstance(msg.tags, dict) else {}
        subtasks = tags.get("subtasks", [])
        name     = tags.get("list_name", msg.content)
        lines    = [f"📋 {name}\n"]
        for s in subtasks:
            prefix = "✓" if s.get("done") else "☐"
            lines.append(f"{prefix} {s['task']}")
        return "\n".join(lines)
