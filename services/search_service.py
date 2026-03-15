"""
Intelligent Search Service — v3
────────────────────────────────────────────────────────────────────────────
KEY CHANGES over v2:
  • _try_list_fetch() — named lists fetched directly, zero LLM, zero hallucination
  • _fetch_todos_direct() — todo queries bypass LLM ranking, pure DB ordered fetch
    (timed todos first by event_time, untimed after)
  • Timestamp delta computed in Python — shown only when results span >1 date
  • Multi-source date detection — groups results by date with subtle labels

FIX (v3.1):
  • IntentService key mapping corrected:
      parsed.get("actions", {}).get("is_query")  instead of parsed.get("intent")
      parsed.get("query") or {}                  instead of parsed.get("entities")
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

from sqlalchemy import and_, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from cerebras_client import CerebrasClient
from database import Category, Message, User

BUCKET_NAMES = ["Remember", "To-Do", "Ideas", "Track", "Events", "List", "Random"]

BUCKET_ALIASES: Dict[str, str] = {
    "todo": "To-Do", "todos": "To-Do", "to-do": "To-Do", "task": "To-Do",
    "tasks": "To-Do", "action": "To-Do", "actions": "To-Do",
    "pending": "To-Do", "due": "To-Do",
    "remember": "Remember", "memory": "Remember", "recall": "Remember",
    "note": "Remember", "saved": "Remember",
    "idea": "Ideas", "ideas": "Ideas", "concept": "Ideas",
    "track": "Track", "log": "Track", "habit": "Track",
    "event": "Events", "events": "Events", "appointment": "Events",
    "meeting": "Events", "schedule": "Events",
}

TODO_KEYWORDS = {
    "todo", "to-do", "to do", "task", "tasks", "pending",
    "checklist", "check list",
}

DUE_DATE_SIGNALS = {
    "due", "todo", "to-do", "task", "pending", "scheduled for",
    "for today", "for tomorrow", "for monday", "for tuesday",
    "for wednesday", "for thursday", "for friday", "for saturday",
    "for sunday", "for this week", "for next week",
}


# ─────────────────────────────────────────────────────────────────────────────
# Date helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_date_range(query: str, ref: date) -> Tuple[Optional[str], Optional[str]]:
    q = query.lower()

    if "yesterday" in q:
        d = ref - timedelta(days=1)
        return str(d), str(d)
    if re.search(r"\btoday\b", q):
        return str(ref), str(ref)
    if "tomorrow" in q:
        d = ref + timedelta(days=1)
        return str(d), str(d)
    if re.search(r"this\s+week", q):
        monday = ref - timedelta(days=ref.weekday())
        return str(monday), str(monday + timedelta(days=6))
    if re.search(r"next\s+week", q):
        monday = ref - timedelta(days=ref.weekday()) + timedelta(weeks=1)
        return str(monday), str(monday + timedelta(days=6))
    if re.search(r"last\s+week", q):
        monday = ref - timedelta(days=ref.weekday() + 7)
        return str(monday), str(monday + timedelta(days=6))
    if re.search(r"this\s+month", q):
        first = ref.replace(day=1)
        last  = (
            ref.replace(month=ref.month + 1, day=1) - timedelta(days=1)
            if ref.month < 12
            else ref.replace(day=31)
        )
        return str(first), str(last)

    m = re.search(r"(?:past|last)\s+(\d+)\s+days?", q)
    if m:
        return str(ref - timedelta(days=int(m.group(1)))), str(ref)

    day_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for i, d in enumerate(day_names):
        if re.search(rf"\b{d}\b", q):
            delta  = (i - ref.weekday()) % 7 or 7
            target = ref + timedelta(days=delta)
            return str(target), str(target)

    MONTHS = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
    for name, num in MONTHS.items():
        if re.search(rf"\b{name}\b", q):
            year  = ref.year if num >= ref.month else ref.year + 1
            first = date(year, num, 1)
            last  = (
                date(year, num + 1, 1) - timedelta(days=1)
                if num < 12
                else date(year, 12, 31)
            )
            return str(first), str(last)

    return None, None


def _detect_bucket(query: str) -> Optional[str]:
    q = query.lower()
    for alias, bucket in BUCKET_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", q):
            return bucket
    return None


def _is_due_date_query(query: str) -> bool:
    q = query.lower()
    return any(sig in q for sig in DUE_DATE_SIGNALS)


def _is_todo_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in TODO_KEYWORDS)


def _extract_person_name(query: str) -> Optional[str]:
    patterns = [
        r"\b(?:about|with|from|call|called|contact)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
    ]
    for p in patterns:
        m = re.search(p, query)
        if m:
            name = m.group(1)
            if name.lower() not in {
                "show", "find", "get", "search", "my", "me",
                "the", "a", "an", "today", "tomorrow",
            }:
                return name
    return None


def _date_label(saved_date: str, today: date) -> str:
    try:
        d     = date.fromisoformat(saved_date)
        delta = (today - d).days
        if delta == 0:  return "today"
        if delta == 1:  return "yesterday"
        if delta <= 6:  return f"{delta}d ago"
        return d.strftime("%d %b")
    except Exception:
        return saved_date


# ─────────────────────────────────────────────────────────────────────────────
# Search service
# ─────────────────────────────────────────────────────────────────────────────

class SearchService:

    def __init__(self, cerebras_client: CerebrasClient):
        self.cerebras = cerebras_client

    async def search(
        self,
        user_phone: str,
        query: str,
        db: AsyncSession,
        limit: int = 15,
        category_filter: Optional[List[str]] = None,
    ) -> Dict:
        user = await self._get_user(user_phone, db)
        if not user:
            return {"results": [], "natural_response": "User not found."}

        today = datetime.utcnow().date()

        # ── 1. Parse intent via IntentService ────────────────────
        from services.intent_service import get_intent_service
        intent_svc = get_intent_service(self.cerebras)
        parsed     = await intent_svc.parse(query, user.name, user.timezone or "Asia/Kolkata")

        # FIX: read from correct keys in the IntentService output schema
        actions      = parsed.get("actions", {})
        is_query     = actions.get("is_query", False)
        query_sub    = parsed.get("query") or {}   # the "query" sub-object, not top-level intent
        list_name_hint = query_sub.get("list_name")
        date_hint      = query_sub.get("date_hint")

        print(f"[search] is_query={is_query} list_hint={list_name_hint!r} for: {query[:60]}")

        # ── 2. Named list query ───────────────────────────────────
        if is_query and list_name_hint:
            list_result = await self._try_list_fetch(
                user.id, query, db, list_name_hint=list_name_hint
            )
            if list_result is not None:
                return list_result

        # Also try list fetch for any query (catches bare "dmart?")
        list_result = await self._try_list_fetch(user.id, query, db)
        if list_result is not None:
            return list_result

        # ── 3. Resolve date range ─────────────────────────────────
        date_from, date_to = _resolve_date_range(query, today)

        # Fall back to IntentService date hint if regex found nothing
        if not date_from and is_query:
            if date_hint == "today":
                date_from = date_to = str(today)
            elif date_hint == "tomorrow":
                d = today + timedelta(days=1)
                date_from = date_to = str(d)
            elif date_hint == "this_week":
                monday = today - timedelta(days=today.weekday())
                date_from = str(monday)
                date_to   = str(monday + timedelta(days=6))

        is_future_q = any(kw in query.lower() for kw in {"upcoming", "future", "all", "everything"})
        is_list_q   = any(kw in query.lower() for kw in TODO_KEYWORDS | {
            "shopping", "shop", "grocery", "buy list", "shopping list"
        })

        # Default undated list/todo queries to today only (not for "all" / "upcoming")
        if not date_from and is_list_q and not is_future_q:
            date_from = str(today)
            date_to   = str(today)

        # ── 4. Direct todo fetch — zero LLM ──────────────────────
        if _is_todo_query(query) or is_query:
            fetch_from  = date_from or str(today)
            fetch_to    = date_to   or str(today)
            todo_result = await self._fetch_todos_direct(user.id, fetch_from, fetch_to, db, limit)
            
            # FIX: always return for todo queries — even if empty.
            # Never fall through to semantic search for explicit todo requests,
            # or List/Random messages will appear via keyword similarity.
            if _is_todo_query(query):
                return todo_result   # returns empty results with correct structure
            
            # For general is_query (non-todo), still allow fallthrough if empty
            if todo_result["results"]:
                return todo_result

        # ── 5. Standard semantic + keyword search ─────────────────
        bucket_hint    = _detect_bucket(query)
        use_due_filter = _is_due_date_query(query) and date_from is not None
        person_hint    = _extract_person_name(query)

        expansion = await self._expand_query(
            query=query, user=user, db=db,
            date_from=date_from, date_to=date_to, bucket_hint=bucket_hint,
        )

        if date_from:
            expansion["date_from"] = date_from
            expansion["date_to"]   = date_to
        if bucket_hint:
            expansion["bucket_filter"] = bucket_hint
        if person_hint:
            expansion.setdefault("entities", [])
            if person_hint not in expansion["entities"]:
                expansion["entities"].insert(0, person_hint)

        messages = await self._retrieve(
            user=user, query=query, expansion=expansion,
            use_due_filter=use_due_filter, db=db, limit=limit * 3,
        )

        ranked = self._rank(
            messages=messages, query=query, expansion=expansion,
            use_due_filter=use_due_filter,
        )[:limit]

        natural = await self._natural_response(
            query=query, results=ranked,
            date_from=date_from, date_to=date_to, today=today,
        )

        return {"results": ranked, "natural_response": natural}

    # ──────────────────────────────────────────────────────────────
    # Direct list fetch — ZERO hallucination
    # ──────────────────────────────────────────────────────────────

    async def _try_list_fetch(
        self, user_id: int, query: str, db: AsyncSession,
        list_name_hint: Optional[str] = None,
    ) -> Optional[Dict]:
        from services.list_service import ListService
        ls = ListService(self.cerebras)

        # Never intercept todo/task queries as list queries
        q_lower = query.lower().strip()
        TODO_BLOCK = {"todo", "to-do", "to do", "task", "tasks", "pending"}
        if any(kw in q_lower for kw in TODO_BLOCK):
            return None

        # Fast path: IntentService already extracted the list name
        if list_name_hint:
            msg = await ls.find_best_matching_list(user_id, list_name_hint, db)
            if msg:
                tags      = msg.tags if isinstance(msg.tags, dict) else {}
                subtasks  = tags.get("subtasks", [])
                list_name = tags.get("list_name", msg.content)
                return {
                    "results": [{
                        "id":           msg.id,
                        "content":      msg.content,
                        "essence":      list_name,
                        "category":     "List",
                        "all_buckets":  ["List"],
                        "priority":     "normal",
                        "tags":         tags,
                        "created_at":   msg.created_at.isoformat(),
                        "due_date":     None,
                        "event_time":   None,
                        "events":       [],
                        "relevance":    100.0,
                        "preview":      f"{len(subtasks)} items",
                        "is_list":      True,
                        "list_message": msg,
                    }],
                    "natural_response": "",
                    "is_list":          True,
                    "list_name":        list_name,
                    "list_message_id":  msg.id,
                }

        intent = await ls.detect_list_intent(query)
        if not intent or intent["intent"] != "show":
            return None

        list_name = intent["list_name"]
        list_type = intent["list_type"]

        msg = await ls.find_best_matching_list(user_id, list_name, db)
        if not msg:
            return {
                "results":          [],
                "natural_response": (
                    f"You don't have a *{list_name}* yet.\n\n"
                    f"Start by sending:\n`{list_name.lower()}:\n- item1\n- item2`"
                ),
                "is_list":   True,
                "list_name": list_name,
                "list_type": list_type,
            }

        tags     = msg.tags if isinstance(msg.tags, dict) else {}
        subtasks = tags.get("subtasks", [])

        return {
            "results": [{
                "id":           msg.id,
                "content":      msg.content,
                "essence":      tags.get("list_name", msg.content),
                "category":     "List",
                "all_buckets":  ["List"],
                "priority":     "normal",
                "tags":         tags,
                "created_at":   msg.created_at.isoformat(),
                "due_date":     None,
                "event_time":   None,
                "events":       [],
                "relevance":    100.0,
                "preview":      f"{len(subtasks)} items",
                "is_list":      True,
                "list_message": msg,
            }],
            "natural_response": "",
            "is_list":          True,
            "list_type":        list_type,
            "list_name":        list_name,
            "list_message_id":  msg.id,
        }

    # ──────────────────────────────────────────────────────────────
    # Direct todo fetch — ZERO LLM
    # ──────────────────────────────────────────────────────────────

    async def _fetch_todos_direct(
        self,
        user_id: int,
        date_from: str,
        date_to: str,
        db: AsyncSession,
        limit: int = 20,
        include_overdue: bool = True,
    ) -> Dict:
        from datetime import date as date_type
        today_str = str(date_type.today())
        is_today_query = (date_from == date_to == today_str)

        # Build the date condition
        # For today queries with include_overdue: also pull past-due undone items
        if include_overdue and is_today_query:
            date_filter = or_(
                # Today's items
                and_(
                    text("messages.tags->>'due_date' >= :df"),
                    text("messages.tags->>'due_date' <= :dt"),
                ),
                # Overdue: has a due_date, it's in the past, not done
                and_(
                    text("messages.tags ? 'due_date'"),
                    text("messages.tags->>'due_date' != ''"),
                    text("messages.tags->>'due_date' < :df"),
                ),
            )
        else:
            date_filter = and_(
                text("messages.tags->>'due_date' >= :df"),
                text("messages.tags->>'due_date' <= :dt"),
            )

        result = await db.execute(
            select(Message, Category)
            .join(Category, Message.category_id == Category.id, isouter=True)
            .where(
                and_(
                    Message.user_id == user_id,
                    date_filter,
                    text("(messages.tags->>'done')::boolean IS NOT TRUE"),
                    text("messages.tags->'all_buckets' @> '\"To-Do\"'::jsonb"),
                    # Exclude pure List messages — primary_bucket = 'List'
                    # Handles both old rows (intent_bucket) and new rows (primary_bucket)
                    text("""
                        COALESCE(
                            messages.tags->>'primary_bucket',
                            messages.tags->>'intent_bucket',
                            'To-Do'
                        ) != 'List'
                    """),
                )
            )
            .params(df=date_from, dt=date_to)
            .order_by(Message.created_at.asc())
            .limit(limit * 2)
        )
        rows = result.all()

        if not rows:
            return {"results": [], "natural_response": "", "is_direct_todo": True}

        timed    = []
        untimed  = []
        overdue  = []

        for message, category in rows:
            tags       = message.tags if isinstance(message.tags, dict) else {}
            evt_time   = tags.get("event_time")
            due        = tags.get("due_date", "")
            is_overdue = bool(due and due < today_str and is_today_query)

            item = {
                "id":          message.id,
                "content":     message.content,
                # Use summary (clean task text) over raw content
                "essence":     message.summary or message.content.split("\n")[0][:80],
                "category":    category.name if category else "To-Do",
                "all_buckets": tags.get("all_buckets", ["To-Do"]),
                "priority":    tags.get("priority", "normal"),
                "tags":        tags,
                "created_at":  message.created_at.isoformat(),
                "due_date":    due,
                "event_time":  evt_time,
                "events":      tags.get("events", []),
                "relevance":   50.0,
                "preview":     message.content[:120],
                "is_overdue":  is_overdue,
            }

            if is_overdue:
                overdue.append(item)
            elif evt_time:
                timed.append(item)
            else:
                untimed.append(item)

        def _dedup(items: List[Dict]) -> List[Dict]:
            seen: set = set()
            out = []
            for item in items:
                key = item["content"].lower().strip()
                if key not in seen:
                    seen.add(key)
                    out.append(item)
            return out

        timed   = _dedup(timed)
        untimed = _dedup(untimed)
        overdue = _dedup(overdue)

        # Sort timed by event_time, overdue by due_date (oldest first)
        timed.sort(key=lambda x: x["event_time"] or "00:00")
        overdue.sort(key=lambda x: x["due_date"] or "")

        # Order: timed today → untimed today → overdue (at bottom, less urgent visually)
        results = (timed + untimed + overdue)[:limit]

        return {
            "results":         results,
            "natural_response": "",
            "is_direct_todo":   True,
            "date_from":        date_from,
            "date_to":          date_to,
            "timed_count":      len(timed),
            "untimed_count":    len(untimed),
            "overdue_count":    len(overdue),
        }

    # ──────────────────────────────────────────────────────────────
    # LLM query expansion
    # ──────────────────────────────────────────────────────────────

    async def _expand_query(
        self, query: str, user: User, db: AsyncSession,
        date_from: Optional[str], date_to: Optional[str], bucket_hint: Optional[str],
    ) -> Dict:
        today_str = datetime.utcnow().strftime("%Y-%m-%d (%A, %d %B %Y)")
        user_cats = await self._get_user_categories(user.id, db)

        prompt = f"""Expand a search query for a personal knowledge base.

USER: {user.name} ({user.occupation})
TODAY: {today_str}
CATEGORIES: {", ".join(user_cats)}
QUERY: "{query}"
PRE-PARSED: date_from={date_from}, date_to={date_to}, bucket={bucket_hint}

Return ONLY this JSON:
{{
  "intent": "find_specific | browse_category | time_based | topic_explore",
  "core_concepts": ["main concepts"],
  "keywords": ["6-10 expanded keywords"],
  "entities": ["people/places/things"],
  "bucket_filter": "{bucket_hint or 'null'}",
  "extra_buckets": ["additional buckets to search"],
  "search_focus": "content | summary | tags | all"
}}"""

        response = await self.cerebras.chat_lite(prompt, max_tokens=400)
        response.setdefault("core_concepts", [])
        response.setdefault("keywords", [])
        response.setdefault("entities", [])
        response.setdefault("intent", "find_specific")
        response.setdefault("search_focus", "all")
        response.setdefault("extra_buckets", [])
        return response

    # ──────────────────────────────────────────────────────────────
    # DB retrieval
    # ──────────────────────────────────────────────────────────────

    async def _retrieve(
        self, user: User, query: str, expansion: Dict,
        use_due_filter: bool, db: AsyncSession, limit: int,
    ) -> List[tuple]:
        semantic_hits: Dict[int, float] = {}
        try:
            from services.embedding_service import embedding_service
            query_embedding = await embedding_service.aembed_query(query)
            embedding_str   = f"[{','.join(map(str, query_embedding))}]"
            sem_sql = text("""
                SELECT m.id, 1 - (m.embedding <=> :emb ::vector) AS similarity
                FROM messages m
                WHERE m.user_id = :uid AND m.embedding IS NOT NULL
                ORDER BY m.embedding <=> :emb ::vector
                LIMIT :lim
            """)
            sem_result    = await db.execute(sem_sql, {"emb": embedding_str, "uid": user.id, "lim": limit})
            semantic_hits = {row.id: float(row.similarity) for row in sem_result}
        except Exception as e:
            print(f"⚠ Semantic search failed: {e}")

        stmt = (
            select(Message, Category)
            .join(Category, Message.category_id == Category.id, isouter=True)
            .where(Message.user_id == user.id)
        )

        bucket_filter  = expansion.get("bucket_filter")
        extra_buckets  = expansion.get("extra_buckets", [])
        if isinstance(extra_buckets, str):
            extra_buckets = [extra_buckets]
        all_target_buckets = list(dict.fromkeys(
            ([bucket_filter] if bucket_filter and bucket_filter in BUCKET_NAMES else [])
            + [b for b in extra_buckets if b in BUCKET_NAMES]
        ))

        if all_target_buckets:
            bucket_conditions = []
            for b in all_target_buckets:
                bucket_conditions.append(Category.name == b)
                bucket_conditions.append(text(f"messages.tags->'all_buckets' @> '\"{b}\"'::jsonb"))
            stmt = stmt.where(or_(*bucket_conditions))

        date_from = expansion.get("date_from")
        date_to   = expansion.get("date_to")
        if date_from and date_to:
            if use_due_filter:
                stmt = stmt.where(
                    or_(
                        and_(
                            text("messages.tags->>'due_date' >= :df"),
                            text("messages.tags->>'due_date' <= :dt"),
                        ),
                        text(
                            "EXISTS (SELECT 1 FROM jsonb_array_elements(messages.tags->'events') ev"
                            " WHERE ev->>'date' >= :df AND ev->>'date' <= :dt)"
                        ),
                    )
                ).params(df=date_from, dt=date_to)
            else:
                stmt = stmt.where(
                    and_(
                        Message.created_at >= datetime.fromisoformat(date_from),
                        Message.created_at < datetime.fromisoformat(date_to) + timedelta(days=1),
                    )
                )

        kw_conds  = []
        all_terms = (
            expansion.get("keywords", [])[:8]
            + expansion.get("core_concepts", [])[:4]
            + expansion.get("entities", [])[:4]
        )
        for term in all_terms:
            t = term.lower()
            kw_conds.append(func.lower(Message.content).contains(t))
            kw_conds.append(func.lower(Message.summary).contains(t))

        if kw_conds:
            stmt = stmt.where(or_(*kw_conds))

        stmt      = stmt.order_by(Message.created_at.desc()).limit(limit)
        kw_result = await db.execute(stmt)
        kw_rows   = kw_result.all()

        kw_ids  = {m.id for m, _ in kw_rows}
        all_ids = kw_ids | set(semantic_hits.keys())
        if not all_ids:
            return []

        final_stmt = (
            select(Message, Category)
            .join(Category, Message.category_id == Category.id, isouter=True)
            .where(Message.id.in_(all_ids))
        )

        if all_target_buckets:
            bucket_conditions = []
            for b in all_target_buckets:
                bucket_conditions.append(Category.name == b)
                bucket_conditions.append(text(f"messages.tags->'all_buckets' @> '\"{b}\"'::jsonb"))
            final_stmt = final_stmt.where(or_(*bucket_conditions))

        if date_from and date_to and use_due_filter:
            final_stmt = final_stmt.where(
                or_(
                    and_(
                        text("messages.tags->>'due_date' >= :df"),
                        text("messages.tags->>'due_date' <= :dt"),
                    ),
                    text(
                        "EXISTS (SELECT 1 FROM jsonb_array_elements(messages.tags->'events') ev"
                        " WHERE ev->>'date' >= :df AND ev->>'date' <= :dt)"
                    ),
                )
            ).params(df=date_from, dt=date_to)

        final_result = await db.execute(final_stmt)
        rows         = final_result.all()

        for message, _ in rows:
            message._semantic_score = semantic_hits.get(message.id, 0.0)

        return list(rows)

    # ──────────────────────────────────────────────────────────────
    # Ranking
    # ──────────────────────────────────────────────────────────────

    def _rank(
        self, messages: List[tuple], query: str,
        expansion: Dict, use_due_filter: bool,
    ) -> List[Dict]:
        scored = []

        for message, category in messages:
            score = self._score(message, category, query, expansion, use_due_filter)
            tags  = message.tags if isinstance(message.tags, dict) else {}

            scored.append({
                "id":          message.id,
                "content":     message.content,
                "essence":     message.summary,
                "category":    category.name if category else "Uncategorized",
                "all_buckets": tags.get("all_buckets", [category.name if category else "?"]),
                "priority":    tags.get("priority", "normal"),
                "tags":        tags,
                "created_at":  message.created_at.isoformat(),
                "due_date":    tags.get("due_date"),
                "event_time":  tags.get("event_time"),
                "events":      tags.get("events", []),
                "relevance":   score,
                "preview":     self._preview(message.content, expansion.get("keywords", [])),
                "_saved_date": message.created_at.strftime("%Y-%m-%d"),
            })

        scored.sort(key=lambda x: (x["relevance"], x["priority"] == "high"), reverse=True)
        return scored

    def _score(
        self, message: Message, category: Optional[Category],
        query: str, expansion: Dict, use_due_filter: bool,
    ) -> float:
        score   = 0.0
        content = message.content.lower()
        summary = (message.summary or "").lower()
        tags    = message.tags if isinstance(message.tags, dict) else {}
        q_lower = query.lower()

        score += getattr(message, "_semantic_score", 0.0) * 20.0
        if q_lower in content:  score += 12.0

        for c in expansion.get("core_concepts", []):
            if c.lower() in content: score += 6.0
            if c.lower() in summary: score += 4.0

        for kw in expansion.get("keywords", []):
            kl = kw.lower()
            if kl in content: score += 2.5
            if kl in summary: score += 1.5

        for entity in expansion.get("entities", []):
            el = entity.lower()
            if el in content: score += 5.0
            if el in str(tags.get("entities", {})).lower(): score += 3.0

        all_msg_buckets = tags.get("all_buckets", [])
        bucket_filter   = expansion.get("bucket_filter")
        extra_buckets   = expansion.get("extra_buckets", [])
        for b in ([bucket_filter] if bucket_filter else []) + (extra_buckets or []):
            if b and ((category and category.name == b) or b in all_msg_buckets):
                score += 8.0

        if use_due_filter:
            date_from = expansion.get("date_from")
            date_to   = expansion.get("date_to")
            if date_from and date_to:
                due = tags.get("due_date")
                if due and date_from <= due <= date_to:
                    score += 15.0
                for ev in tags.get("events", []):
                    if isinstance(ev, dict):
                        ev_date = ev.get("date", "")
                        if ev_date and date_from <= ev_date <= date_to:
                            score += 15.0

        if tags.get("priority") == "high":  score += 2.0
        age = (datetime.utcnow() - message.created_at).days
        if age < 1:   score += 2.0
        elif age < 7: score += 1.0

        return score

    # ──────────────────────────────────────────────────────────────
    # Natural response
    # ──────────────────────────────────────────────────────────────

    async def _natural_response(
        self, query: str, results: List[Dict],
        date_from: Optional[str], date_to: Optional[str], today: date,
    ) -> str:
        if not results:
            time_ctx = ""
            if date_from:
                if date_from == date_to:
                    d        = datetime.fromisoformat(date_from).strftime("%A, %d %b")
                    time_ctx = f" for {d}"
                else:
                    time_ctx = f" between {date_from} and {date_to}"
            return f"Nothing found{time_ctx} in your notes."

        saved_dates  = {r["_saved_date"] for r in results if r.get("_saved_date")}
        multi_source = len(saved_dates) > 1
        today_str    = str(today)
        yesterday    = str(today - timedelta(days=1))

        if date_from and date_to:
            if date_from == date_to == today_str:    time_ctx = "today"
            elif date_from == date_to == yesterday:   time_ctx = "yesterday"
            elif date_from == date_to:                time_ctx = datetime.fromisoformat(date_from).strftime("%A, %d %b")
            else:                                     time_ctx = f"between {date_from} and {date_to}"
        else:
            time_ctx = "any time"

        context = ""
        for r in results[:8]:
            due       = f" [due: {r['due_date']}]" if r.get("due_date") else ""
            ev_time   = f" at {r['event_time']}" if r.get("event_time") else ""
            buckets   = ", ".join(r.get("all_buckets", [r.get("category", "?")]))
            date_note = (
                f" [saved {_date_label(r.get('_saved_date', ''), today)}]"
                if multi_source else ""
            )
            context += f"\n[{buckets}]{due}{ev_time}{date_note}: {r['content']}\n"

        prompt = f"""Personal assistant answering a search in the user's notes.

TODAY: {today}
USER ASKED: "{query}"
TIME SCOPE: {time_ctx}

MATCHING NOTES:
{context}

Instructions:
- Answer the specific question directly
- List tasks/events clearly with times if available
- Be concise — max 4 sentences or a short list
- If results are from different dates, mention dates naturally (e.g. "yesterday you saved...")
- Never invent information not in the notes
- Don't say "based on your notes"

Reply:"""

        return await self.cerebras._chat_completion(prompt, max_tokens=300, temperature=0.2)

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    def _preview(self, content: str, keywords: List[str]) -> str:
        if len(content) <= 150:
            return content
        lo = content.lower()
        for kw in keywords:
            idx = lo.find(kw.lower())
            if idx != -1:
                start = max(0, idx - 50)
                end   = min(len(content), idx + 100)
                pre   = content[start:end]
                if start > 0:           pre = "…" + pre
                if end < len(content):  pre = pre + "…"
                return pre
        return content[:150] + "…"

    async def _get_user(self, phone: str, db: AsyncSession) -> Optional[User]:
        result = await db.execute(select(User).where(User.phone_number == phone))
        return result.scalar_one_or_none()

    async def _get_user_categories(self, user_id: int, db: AsyncSession) -> List[str]:
        result = await db.execute(select(Category.name).where(Category.user_id == user_id))
        return [row[0] for row in result.all()]
