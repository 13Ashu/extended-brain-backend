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
from models import MessageType as MT


async def _save_list_embedding(message_id: int, list_name: str, items: List[str], db: AsyncSession):
    """Generate and store embedding for a list message so it's findable via semantic search."""
    try:
        from services.embedding_service import embedding_service
        enriched = list_name + " " + " ".join(items)
        embedding = await embedding_service.aembed(enriched.strip(), task_type="RETRIEVAL_DOCUMENT")
        await db.execute(update(Message).where(Message.id == message_id).values(embedding=embedding))
        await db.commit()
    except Exception as e:
        print(f"⚠ List embedding failed (non-critical): {e}")


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

# Words/phrases that flag a name as a neutral time/category label (NOT a named list)
_BLOCKED_NAME_WORDS = {
    "todo", "to-do", "to do", "task", "tasks", "remind", "reminder",
    "update", "notes", "note", "things", "items",
    "today", "tomorrow", "morning", "evening", "night",
    "urgent", "important", "priority", "agenda",
}

# Signals that confirm a name refers to a list-like thing
LIST_SIGNALS = {
    # Generic
    "list", "lists", "bag", "checklist",
    # Shopping
    "shopping", "grocery", "groceries", "supermarket", "store",
    # Stores / delivery apps
    "dmart", "mall", "zepto", "blinkit", "swiggy", "amazon",
    "flipkart", "bigbasket", "instamart", "dunzo",
    # Packing / travel
    "packing", "pack", "travel", "trip", "luggage", "vacation",
    # Media
    "movies", "movie", "watch", "watching", "shows", "series",
    # Reading
    "reading", "books", "book",
}

# Patterns that indicate SHOW intent
LIST_SHOW_PATTERNS = [
    # "show my dmart list today" / "get shopping list" / "give mall list"
    r"(?:show|get|give|display|open)\s+(?:my\s+|the\s+)?(\w[\w\s\-]+?(?:list|bag))(?:\s+today|\s+now)?\s*\??$",
    # "my dmart shopping list?" / "my bag list today?"
    r"^(?:my\s+)?(\w[\w\s\-]+?(?:list|bag))(?:\s+today|\s+now)?\s*\?+\s*$",
    # "whats in my shopping list" / "what is in my bag list"
    r"what(?:\'s| is) (?:in|on) (?:my\s+|the\s+)?([\w][\w\s\-]+?(?:list|bag))",
    # bare "dmart list?" or "movie list?"
    r"^(\w[\w\s\-]+?(?:list|bag))(?:\s+today|\s+now)?\s*\?+\s*$",
    # "show me my X list"
    r"show\s+me\s+(?:my\s+|the\s+)?(\w[\w\s\-]+?(?:list|bag))",
]


def _name_blocked(name: str) -> bool:
    """Return True if name looks like a neutral time/category label, not a named list."""
    words = set(name.lower().split())
    return bool(words & _BLOCKED_NAME_WORDS) or "to-do" in name or "to do" in name


def _has_list_signal(text: str) -> bool:
    """Return True if text contains a word that strongly suggests a list context."""
    return any(sig in text for sig in LIST_SIGNALS)

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
        if line and not line.endswith(":"):
            if "," in line:
                # comma-separated items on one line: "apple, banana, strawberry"
                parts = [p.strip() for p in line.split(",") if p.strip()]
                items.extend(parts)
            elif len(line.split()) <= 8:
                items.append(line)

    # Fallback: inline add — "add pen, mug and glue to list" / "add pen and mug to dmart"
    if not items:
        m = re.search(r"(?:add|put)\s+(.+?)\s+(?:to|in|into)\b", content.lower())
        if m:
            raw = m.group(1).strip()
            # Split on comma, semicolon, or " and "
            parts = re.split(r"[,;]|\band\b", raw)
            items = [p.strip() for p in parts if p.strip()]

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
        TODO_BLOCK = {"todo", "to-do", "to do", "task", "tasks", "pending"}
        if any(kw in lc for kw in TODO_BLOCK):
            return None

        # Also catch "add X to/in <name>" patterns even without list keyword
        _add_pattern = (
            re.search(r"\badd\b.{1,60}\bto\b", lc) or
            re.search(r"\bput\b.{1,60}\bin\b", lc)
        )
        if not _has_list_signal(lc) and not _add_pattern:
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
            "1. create_or_add — user is creating a named list OR adding items to an existing one\n"
            "2. show — user wants to see/retrieve a named list\n"
            "3. none — regular note, todo, reminder, search, anything else\n\n"
            "RULES:\n"
            "- A named list has a NAME and ITEMS (bullets, numbered, or comma-separated inline)\n"
            "- The list_name MUST preserve the context word: dmart→Dmart, mall→Mall, japan→Japan trip\n"
            "- For show intent: infer full list name (dmart list → Dmart Shopping List)\n"
            "- todo batch with action items is NOT a list → none\n"
            "- reminders, notes, searches are NOT lists → none\n"
            "- Extract ALL items; never include the header line in items\n"
            "- For inline adds ('add eggs and milk to dmart'), items = ['eggs', 'milk']\n"
            "- Split items on comma, semicolon, and the word 'and'\n\n"
            "EXAMPLES (study carefully):\n\n"
            "Input: dmart shopping:\n- Pen\n- coffee mug\n- glue\n"
            'Output: {"intent":"create_or_add","list_name":"Dmart Shopping List","list_type":"shopping","items":["Pen","coffee mug","glue"]}\n\n'
            "Input: My shopping list for mall:\n- party shirt\n- brown belt\n"
            'Output: {"intent":"create_or_add","list_name":"Mall Shopping List","list_type":"shopping","items":["party shirt","brown belt"]}\n\n'
            "Input: Bag list for exam:\n- pencil\n- eraser\n- graph book\n"
            'Output: {"intent":"create_or_add","list_name":"Exam Bag List","list_type":"bag","items":["pencil","eraser","graph book"]}\n\n'
            "Input: add to dmart list: eggs, milk, bread\n"
            'Output: {"intent":"create_or_add","list_name":"Dmart Shopping List","list_type":"shopping","items":["eggs","milk","bread"]}\n\n'
            "Input: add eggs and milk to dmart\n"
            'Output: {"intent":"create_or_add","list_name":"Dmart Shopping List","list_type":"shopping","items":["eggs","milk"]}\n\n'
            "Input: add pasta, sauce and cheese to my grocery list\n"
            'Output: {"intent":"create_or_add","list_name":"Grocery List","list_type":"shopping","items":["pasta","sauce","cheese"]}\n\n'
            "Input: can you add sunscreen and sandals to japan trip packing list\n"
            'Output: {"intent":"create_or_add","list_name":"Japan Trip Packing List","list_type":"packing","items":["sunscreen","sandals"]}\n\n'
            "Input: put shampoo and soap in my dmart shopping list\n"
            'Output: {"intent":"create_or_add","list_name":"Dmart Shopping List","list_type":"shopping","items":["shampoo","soap"]}\n\n'
            "Input: Japan trip:\n- book flights\n- get visa\n- hotel\n"
            'Output: {"intent":"create_or_add","list_name":"Japan Trip List","list_type":"packing","items":["book flights","get visa","hotel"]}\n\n'
            "Input: movies to watch:\n- Interstellar\n- Dune\n- Oppenheimer\n"
            'Output: {"intent":"create_or_add","list_name":"Movies To Watch List","list_type":"watching","items":["Interstellar","Dune","Oppenheimer"]}\n\n'
            "Input: Show dmart list?\n"
            'Output: {"intent":"show","list_name":"Dmart Shopping List","list_type":"shopping","items":[]}\n\n'
            "Input: show my mall shopping list\n"
            'Output: {"intent":"show","list_name":"Mall Shopping List","list_type":"shopping","items":[]}\n\n'
            "Input: what's in my bag list?\n"
            'Output: {"intent":"show","list_name":"Bag List","list_type":"bag","items":[]}\n\n'
            "Input: get my exam bag list\n"
            'Output: {"intent":"show","list_name":"Exam Bag List","list_type":"bag","items":[]}\n\n'
            "Input: show me my grocery list\n"
            'Output: {"intent":"show","list_name":"Grocery List","list_type":"shopping","items":[]}\n\n'
            "Input: todo for today:\n- call mom\n- buy milk\n"
            'Output: {"intent":"none","list_name":null,"list_type":null,"items":[]}\n\n'
            "Input: remind me to drink water every 2 hours\n"
            'Output: {"intent":"none","list_name":null,"list_type":null,"items":[]}\n\n'
            "Input: buy milk tomorrow\n"
            'Output: {"intent":"none","list_name":null,"list_type":null,"items":[]}\n\n'
            "Input: search my grocery notes\n"
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
        # For create_or_add: if LLM found no items but content has inline items,
        # try extracting them ourselves rather than giving up
        if intent == "create_or_add" and not items:
            items = _extract_items_from_content(content)
            if not items:
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
        """
        Pure-regex list detection. No LLM. Handles all common user formats:
          1. "add ITEMS to LIST"               add eggs and milk to dmart
          2. "add to LIST: ITEMS"              add to dmart list: eggs, milk
          3. "add to LIST\\n- items"           add to grocery:\\n- butter
          4. "NAME:\\n items"  (multi-line)    dmart:\\n- apple\\n- milk
          5. "NAME\\n- items"  (no colon)      Mall shopping list\\n- shirt
          6. "NAME: item1, item2"  (inline)    dmart: apple, milk, bread
          7. show queries                      show my grocery list
        """
        lc = content.lower().strip()

        def _make(intent: str, raw_name: str, items: List[str]) -> Dict:
            list_name = _normalize_list_name(raw_name)
            return {
                "intent":    intent,
                "list_name": list_name,
                "list_type": _classify_list_type(list_name),
                "items":     items,
            }

        def _split(s: str) -> List[str]:
            parts = re.split(r"[,;]|\band\b", s)
            return [p.strip() for p in parts if 2 <= len(p.strip()) < 100]

        # ── SHOW ──────────────────────────────────────────────────
        for pattern in LIST_SHOW_PATTERNS:
            m = re.search(pattern, lc)
            if m:
                raw_name = m.group(1).strip()
                if _name_blocked(raw_name):
                    continue
                meaningful = [
                    w for w in raw_name.split()
                    if w not in SKIP_WORDS and len(w) > 1 and w not in {"list", "bag"}
                ]
                if not meaningful:
                    continue
                return _make("show", raw_name, [])

        # ── ADD: "add/put ITEMS to/in LIST" ───────────────────────
        # "add eggs and milk to dmart"
        # "can you add sunscreen and sandals to japan packing list"
        # "also add toothpaste to the dmart list"
        m = re.match(
            r"(?:(?:can\s+you|please|also|just)\s+)?"
            r"(?:add|put)\s+"
            r"(?!(?:to|in|into)\s)"          # exclude "add to X" handled below
            r"(.+?)\s+(?:to|in|into)\s+"
            r"(?:the\s+|my\s+)?([\w][\w\s\-\']+?)(?:\s*$|[.?!,\n])",
            lc,
        )
        if m:
            raw_items_str = m.group(1).strip()
            raw_name      = m.group(2).strip()
            if not _name_blocked(raw_name) and _has_list_signal(raw_name):
                items = _split(raw_items_str)
                if items:
                    return _make("create_or_add", raw_name, items)

        # ── ADD: "add to/in LIST: ITEMS" or "add to LIST\\n- items" ──
        # "add to dmart list: eggs, milk, bread"
        # "add in grocery list:\\n- butter\\n- jam"
        m = re.match(
            r"(?:add|put)\s+(?:to|in|into)\s+"
            r"(?:the\s+|my\s+)?([\w][\w\s\-]+?)[\s:,]+(.+)$",
            lc, re.DOTALL,
        )
        if m:
            raw_name  = m.group(1).strip()
            remainder = m.group(2).strip()
            if not _name_blocked(raw_name):
                first_line = remainder.split("\n")[0].strip()
                items = _split(first_line) if first_line else []
                if not items:
                    items = _extract_items_from_content(f"{raw_name}:\n{remainder}")
                if items:
                    return _make("create_or_add", raw_name, items)

        # ── CREATE: "NAME:\\n items" (multi-line, any item style) ─
        # "dmart:\\n- apple\\n- milk"
        # "grocery list:\\napple, milk, bread" (bare comma line)
        # "Japan trip:\\n- book flights\\n- get visa"
        m = re.search(r"^([\w][\w\s\-]+?):\s*\n(.+)", content, re.MULTILINE | re.DOTALL)
        if m:
            raw_name = m.group(1).strip().lower()
            if not _name_blocked(raw_name):
                items = _extract_items_from_content(content)
                if items:
                    has_bullets = bool(re.search(r"\n\s*[-*•]|\n\s*\d+[.)]", content))
                    if _has_list_signal(raw_name) or has_bullets:
                        return _make("create_or_add", raw_name, items)

        # ── CREATE: "NAME\\n- items" (no colon, needs list signal) ─
        # "Mall shopping list\\n- party shirt\\n- brown belt"
        # "Packing list\\n1. passport\\n2. charger"
        m = re.search(r"^([\w][\w\s\-]+?)\s*\n\s*[-*•\d]", content, re.MULTILINE)
        if m:
            raw_name = m.group(1).strip().lower()
            if not _name_blocked(raw_name) and _has_list_signal(raw_name):
                items = _extract_items_from_content(content)
                if items:
                    return _make("create_or_add", raw_name, items)

        # ── CREATE: inline colon "NAME: item1, item2" (single line) ─
        # "dmart: apple, milk, bread"
        # "grocery list: eggs, butter, cheese"
        # "movies to watch: interstellar, dune, oppenheimer"
        # "zepto: soap, shampoo, toothpaste"
        # Require list signal + ≥2 items to avoid "note: call mom" false positives
        m = re.match(r"^([\w][\w\s\-]+?):\s*([^:\n]{5,})$", lc)
        if m:
            raw_name      = m.group(1).strip()
            raw_items_str = m.group(2).strip()
            if not _name_blocked(raw_name) and _has_list_signal(raw_name):
                items = _split(raw_items_str)
                if len(items) >= 2:
                    return _make("create_or_add", raw_name, items)

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
                    text("(messages.tags->>'is_list')::boolean = true"),
                    text("(messages.tags->>'is_active')::boolean IS NOT FALSE"),
                )
            )
            .params(name=list_name)
            .order_by(Message.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def find_best_matching_list(
        self, user_id: int, list_name: str, db: AsyncSession,
        group_id: Optional[int] = None,
    ) -> Optional[Message]:
        """
        Find best matching list — group-scoped when group_id is set, else personal.
        Exact name match first, then fuzzy by key words.
        """
        # Determine the scope filter: group messages OR personal messages
        scope_filter = (Message.group_id == group_id) if group_id else (
            and_(Message.user_id == user_id, Message.group_id.is_(None))
        )

        # Exact match
        result = await db.execute(
            select(Message)
            .where(and_(
                scope_filter,
                text("messages.tags->>'list_name' ILIKE :name"),
                text("(messages.tags->>'is_list')::boolean = true"),
                text("(messages.tags->>'is_active')::boolean IS NOT FALSE"),
            ))
            .params(name=list_name)
            .order_by(Message.created_at.desc())
            .limit(1)
        )
        msg = result.scalar_one_or_none()
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
                            scope_filter,
                            text("messages.tags->>'list_type' = :lt"),
                            text("(messages.tags->>'is_list')::boolean = true"),
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
                    scope_filter,
                    text("(messages.tags->>'is_list')::boolean = true"),
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

        def score(m: Message) -> int:
            stored = (m.tags or {}).get("list_name", "").lower()
            return sum(1 for kw in key_words if kw in stored)

        candidates.sort(key=score, reverse=True)
        return candidates[0]

    async def create_or_add(
        self,
        user_id: int,
        list_name: str,
        list_type: str,
        items: List[str],
        db: AsyncSession,
        group_id: Optional[int] = None,
        due_date: Optional[str] = None,
        bucket: str = "Remember",
    ) -> Tuple[Message, int, bool]:
        """
        Create if not exists, add items if exists.
        When group_id is provided, the list is shared across the group.
        Returns (message, items_added, was_created).
        """
        msg = await self.find_best_matching_list(user_id, list_name, db, group_id=group_id)

        if not msg:
            msg = await self._create_list(user_id, list_name, list_type, items, db,
                                          group_id=group_id, due_date=due_date, bucket=bucket)
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
        # Update due_date only if a new one is supplied and none exists yet
        if due_date and not tags.get("due_date"):
            tags["due_date"] = due_date

        await db.execute(
            update(Message).where(Message.id == msg.id).values(tags=tags)
        )
        await db.commit()
        await db.refresh(msg)

        # Refresh embedding to include new items
        all_items = [s["task"] for s in existing]
        await _save_list_embedding(msg.id, tags.get("list_name", msg.content), all_items, db)

        return msg, added, False

    async def _create_list(
        self, user_id: int, list_name: str, list_type: str,
        items: List[str], db: AsyncSession,
        group_id: Optional[int] = None,
        due_date: Optional[str] = None,
        bucket: str = "Remember",
    ) -> Message:
        cat = await db.scalar(
            select(Category).where(
                and_(Category.user_id == user_id, Category.name == bucket)
            )
        )
        if not cat:
            cat = Category(user_id=user_id, name=bucket, description="Named lists")
            db.add(cat)
            await db.flush()

        now      = datetime.utcnow()
        subtasks = [{"task": item, "done": False, "added_at": now.isoformat()} for item in items]

        tags: dict = {
            "list_type":      list_type,
            "list_name":      list_name,
            "subtasks":       subtasks,
            "all_buckets":    [bucket],
            "primary_bucket": bucket,
            "is_list":        True,
            "is_active":      True,
            "item_count":     len(items),
        }
        if due_date:
            tags["due_date"] = due_date

        msg = Message(
            user_id=user_id,
            group_id=group_id,
            category_id=cat.id,
            content=list_name,
            message_type=MT("text"),
            summary=f"{list_name} — {len(items)} items",
            tags=tags,
            created_at=now,
        )
        db.add(msg)
        await db.commit()
        await db.refresh(msg)
        await _save_list_embedding(msg.id, list_name, items, db)
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

    async def add_item(self, message_id: int, task: str) -> Optional[int]:
        """Append a new item to the list. Returns the new item's index, or None on failure."""
        async with async_session_maker() as session:
            msg = await session.scalar(select(Message).where(Message.id == message_id))
            if not msg:
                return None
            tags     = dict(msg.tags or {})
            subtasks = list(tags.get("subtasks", []))
            subtasks.append({"task": task, "done": False, "added_at": datetime.utcnow().isoformat()})
            tags["subtasks"]   = subtasks
            tags["item_count"] = len(subtasks)
            await session.execute(
                update(Message).where(Message.id == message_id).values(tags=tags)
            )
            await session.commit()
        return len(subtasks) - 1

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
                    text("(messages.tags->>'is_list')::boolean = true"),
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
