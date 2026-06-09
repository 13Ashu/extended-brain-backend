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
        group_id: Optional[int] = None,
        fast: bool = False,
    ) -> Dict:
        if not query or not query.strip():
            return {"results": [], "natural_response": "Please enter a search query."}

        user = await self._get_user(user_phone, db)
        if not user:
            return {"results": [], "natural_response": "User not found."}

        from services import redis_cache

        # ── Cache check (skip for group searches — group data changes frequently) ──
        if not group_id:
            ck = redis_cache.search_key(user.id, query)
            cached = await redis_cache.cache_get(ck)
            if cached:
                return cached
        else:
            ck = None

        today = datetime.utcnow().date()

        # ── 0. Image search — highest priority ───────────────────────
        if query.lower().strip().startswith("image:"):
            return await self._fetch_images_direct(user.id, query, db, limit)

        import time as _time
        _t0 = _time.monotonic()

        # ── TIER 1: keyword-only, no embedding (fast=True) ───────────
        # iOS calls this first and shows the result immediately.
        # Text match: content/summary ILIKE — returns in ~5ms.
        date_from, date_to = _resolve_date_range(query, today)
        bucket_hint        = _detect_bucket(query)
        words              = [w for w in query.lower().split() if len(w) > 2]
        base_expansion = {
            "core_concepts": words,
            "keywords":      words,
            "entities":      [],
            "intent":        "find_specific",
            "search_focus":  "all",
            "extra_buckets": [],
            "bucket_filter": bucket_hint,
        }
        if date_from:
            base_expansion["date_from"] = date_from
            base_expansion["date_to"]   = date_to

        if fast:
            messages = await self._retrieve(
                user=user, query=query, expansion=base_expansion,
                use_due_filter=bool(date_from), db=db, limit=limit * 2,
                group_id=group_id, skip_semantic=True,
            )
            ranked = self._rank(messages=messages, query=query, expansion=base_expansion,
                                use_due_filter=bool(date_from))[:limit]
            print(f"[search] tier1 keyword → {_time.monotonic()-_t0:.2f}s ({len(ranked)} results)")
            result = {"results": ranked, "natural_response": ""}
            if ck:
                await redis_cache.cache_set(ck, result, ex=120)
            return result

        # ── TIERS 2+3: embedding + optional LLM expansion (fast=False) ─
        # iOS calls this after showing the tier-1 result. Runs in parallel:
        #   Tier 2 — embed raw query → vector similarity
        #   Tier 3 — LLM expand query → richer keyword terms (skipped for ≤3-word queries)
        from services.embedding_service import embedding_service
        import asyncio as _asyncio

        # Todo queries: direct DB fetch, no embeddings needed
        if _is_todo_query(query):
            use_due = bool(date_from)
            fetch_from = date_from or str(today)
            fetch_to   = date_to   or str(today)
            todo_result = await self._fetch_todos_direct(user.id, fetch_from, fetch_to, db, limit)
            print(f"[search] todo_direct → {_time.monotonic()-_t0:.2f}s")
            return todo_result

        use_due_filter = _is_due_date_query(query) and date_from is not None
        person_hint    = _extract_person_name(query)

        # Tier 2 and Tier 3 start in parallel
        embed_task  = _asyncio.create_task(embedding_service.aembed_query(query))
        expand_task = (
            _asyncio.create_task(
                self._expand_query(query=query, user=user, db=db,
                                   date_from=date_from, date_to=date_to, bucket_hint=bucket_hint)
            )
            if len(words) > 3 else None
        )

        try:
            query_embedding = await embed_task
        except Exception as e:
            print(f"[search] embed failed: {e}")
            query_embedding = None

        expansion = base_expansion
        if expand_task:
            try:
                expansion = await expand_task
                if date_from:
                    expansion["date_from"] = date_from
                    expansion["date_to"]   = date_to
                if bucket_hint:
                    expansion["bucket_filter"] = bucket_hint
            except Exception as e:
                print(f"[search] expand failed: {e}")

        if person_hint:
            expansion.setdefault("entities", [])
            if person_hint not in expansion["entities"]:
                expansion["entities"].insert(0, person_hint)

        print(f"[search] tier2+3 ready → {_time.monotonic()-_t0:.2f}s")

        messages = await self._retrieve(
            user=user, query=query, expansion=expansion,
            use_due_filter=use_due_filter, db=db, limit=limit * 3,
            group_id=group_id, precomputed_embedding=query_embedding,
        )
        ranked = self._rank(
            messages=messages, query=query, expansion=expansion,
            use_due_filter=use_due_filter,
        )[:limit]

        print(f"[search] tier2+3 done → {_time.monotonic()-_t0:.2f}s ({len(ranked)} results)")
        result = {"results": ranked, "natural_response": ""}
        if ck:
            await redis_cache.cache_set(ck, result, ex=300)
        return result


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

        print(f"[todos_direct] user={user_id} df={date_from} dt={date_to} is_today={is_today_query}")

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
                "split_from":  tags.get("split_from", ""),
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
        words = [w for w in query.lower().split() if len(w) > 2]

        # Short queries (≤ 3 meaningful words) — embedding handles semantics, skip LLM
        if len(words) <= 3:
            return {
                "core_concepts": words,
                "keywords":      words,
                "entities":      [],
                "intent":        "find_specific",
                "search_focus":  "all",
                "extra_buckets": [],
                "bucket_filter": bucket_hint,
            }

        today_str = datetime.utcnow().strftime("%Y-%m-%d (%A, %d %B %Y)")
        user_cats = await self._get_user_categories(user.id, db)

        prompt = f"""Expand a search query for a personal knowledge base.

USER: {user.name}
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

        try:
            response = await self.cerebras.chat_lite(prompt, max_tokens=300)
        except Exception as e:
            print(f"[search] _expand_query failed ({type(e).__name__}): {e}")
            response = {}
        response.setdefault("core_concepts", words)
        response.setdefault("keywords", words)
        response.setdefault("entities", [])
        response.setdefault("intent", "find_specific")
        response.setdefault("search_focus", "all")
        response.setdefault("extra_buckets", [])
        if bucket_hint and not response.get("bucket_filter"):
            response["bucket_filter"] = bucket_hint
        return response

    # ──────────────────────────────────────────────────────────────
    # DB retrieval
    # ──────────────────────────────────────────────────────────────

    async def _retrieve(
        self, user: User, query: str, expansion: Dict,
        use_due_filter: bool, db: AsyncSession, limit: int,
        group_id: Optional[int] = None,
        precomputed_embedding: Optional[List[float]] = None,
        skip_semantic: bool = False,
    ) -> List[tuple]:
        semantic_hits: Dict[int, float] = {}
        # Only include messages with cosine similarity above this floor.
        # Gemini text-embedding-004 (1536-dim): short unrelated texts still score 0.30–0.38.
        # 0.40 is the practical cutoff for "genuinely related" in this embedding space.
        MIN_SEMANTIC_SIMILARITY = 0.40
        try:
            if skip_semantic:
                raise ValueError("semantic skipped")
            from services.embedding_service import embedding_service
            query_embedding = precomputed_embedding or await embedding_service.aembed_query(query)
            embedding_str   = f"[{','.join(map(str, query_embedding))}]"
            if group_id:
                sem_sql = text("""
                    SELECT m.id, 1 - (m.embedding <=> :emb ::vector) AS similarity
                    FROM messages m
                    WHERE m.group_id = :gid AND m.embedding IS NOT NULL
                      AND (1 - (m.embedding <=> :emb ::vector)) > :min_sim
                    ORDER BY m.embedding <=> :emb ::vector
                    LIMIT :lim
                """)
                sem_result = await db.execute(sem_sql, {"emb": embedding_str, "gid": group_id, "lim": limit, "min_sim": MIN_SEMANTIC_SIMILARITY})
            else:
                sem_sql = text("""
                    SELECT m.id, 1 - (m.embedding <=> :emb ::vector) AS similarity
                    FROM messages m
                    WHERE m.user_id = :uid AND m.group_id IS NULL AND m.embedding IS NOT NULL
                      AND (1 - (m.embedding <=> :emb ::vector)) > :min_sim
                    ORDER BY m.embedding <=> :emb ::vector
                    LIMIT :lim
                """)
                sem_result = await db.execute(sem_sql, {"emb": embedding_str, "uid": user.id, "lim": limit, "min_sim": MIN_SEMANTIC_SIMILARITY})
            semantic_hits = {row.id: float(row.similarity) for row in sem_result}
            # Relative filter: drop candidates more than 0.12 below the best score.
            # Prevents weak-but-above-floor matches from polluting results when one
            # result is clearly dominant (e.g. "soup recipe" → 0.663 vs 0.503/0.494).
            # Topic queries with clustered results (e.g. 0.72/0.68/0.65) all pass.
            if semantic_hits:
                max_sim = max(semantic_hits.values())
                RELATIVE_GAP = 0.12
                semantic_hits = {mid: sim for mid, sim in semantic_hits.items()
                                 if sim >= max_sim - RELATIVE_GAP}
                scores = sorted(semantic_hits.values(), reverse=True)
                print(f"[search] semantic hits={len(scores)} scores={[round(s,3) for s in scores[:5]]} (floor={MIN_SEMANTIC_SIMILARITY} gap={RELATIVE_GAP})")
        except Exception as e:
            print(f"⚠ Semantic search failed: {e}")

        stmt = (
            select(Message, Category)
            .join(Category, Message.category_id == Category.id, isouter=True)
        )
        if group_id:
            stmt = stmt.where(Message.group_id == group_id)
        else:
            stmt = stmt.where(and_(Message.user_id == user.id, Message.group_id.is_(None)))

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
        # Raw query words always searched first — zero LLM dependency, instant match
        raw_query_words = [w for w in query.lower().split() if len(w) > 2]
        for i, w in enumerate(raw_query_words):
            pattern = f"%{w}%"
            kw_conds.append(func.lower(Message.content).contains(w))
            kw_conds.append(func.lower(Message.summary).contains(w))
            # Search list item text (tags.subtasks[].task) — "handcream" finds its list
            kw_conds.append(
                text(
                    "EXISTS (SELECT 1 FROM jsonb_array_elements(messages.tags->'subtasks') sub"
                    f" WHERE lower(sub->>'task') LIKE :rw_sub_{i})"
                ).bindparams(**{f"rw_sub_{i}": pattern})
            )
            # Search original multi-task dump text (tags.split_from)
            kw_conds.append(
                text(f"lower(messages.tags->>'split_from') LIKE :rw_sf_{i}")
                .bindparams(**{f"rw_sf_{i}": pattern})
            )
            # Search original single-task dump text (tags.original_dump)
            kw_conds.append(
                text(f"lower(messages.tags->>'original_dump') LIKE :rw_od_{i}")
                .bindparams(**{f"rw_od_{i}": pattern})
            )
        # Then LLM-expanded terms (may overlap, OR logic handles dedup)
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

            # For list messages, always surface items as a top-level field so the
            # iOS itemsDirect field is populated. Check is_list flag (not legacy "List" bucket).
            is_list = bool(tags.get("is_list", False))
            list_items = None
            if is_list:
                subtasks   = tags.get("subtasks", [])
                list_items = [
                    {"task": s["task"], "done": s.get("done", False)}
                    for s in subtasks if isinstance(s, dict) and "task" in s
                ]

            scored.append({
                "id":           message.id,
                "content":      message.content,
                "essence":      message.summary,
                "message_type": message.message_type.value if message.message_type else "text",
                "media_url":    message.media_url,
                "category":     category.name if category else "Uncategorized",
                "all_buckets":  tags.get("all_buckets", [category.name if category else "?"]),
                "priority":     tags.get("priority", "normal"),
                "tags":         tags,
                "created_at":   message.created_at.isoformat(),
                "due_date":     tags.get("due_date"),
                "event_time":   tags.get("event_time"),
                "events":       tags.get("events", []),
                "is_list":      is_list,
                "is_done":      bool(tags.get("done", False)),
                "relevance":    score,
                "preview":      self._preview(message.content, expansion.get("keywords", [])),
                "_saved_date":  message.created_at.strftime("%Y-%m-%d"),
                **({"items": list_items} if list_items is not None else {}),
            })

        scored.sort(key=lambda x: (x["relevance"], x["priority"] == "high"), reverse=True)

        # Drop results with negligible relevance — pure semantic noise with no keyword overlap.
        # A result that matched only via weak vector similarity scores ≤ ~6; any keyword hit
        # adds at minimum 8–12 points. This threshold removes semantic stragglers.
        MIN_RELEVANCE = 8.0
        if any(s["relevance"] >= MIN_RELEVANCE for s in scored):
            scored = [s for s in scored if s["relevance"] >= MIN_RELEVANCE]

        return scored

    def _score(
        self, message: Message, category: Optional[Category],
        query: str, expansion: Dict, use_due_filter: bool,
    ) -> float:
        from rapidfuzz import fuzz

        score   = 0.0
        content = message.content.lower()
        summary = (message.summary or "").lower()
        tags    = message.tags if isinstance(message.tags, dict) else {}
        q_lower = query.lower()

        # Semantic score (secondary to text)
        score += getattr(message, "_semantic_score", 0.0) * 15.0

        # Primary text score via token_set_ratio:
        # returns 100 when all query words appear in the stored text (any order, any extras).
        # "people call" → "people to call" = 100 (same as exact match)
        # "peoples call" → "people to call" = 84 (fuzzy on typo)
        # Take max(content, summary) to avoid double-counting.
        text_tsr = max(
            fuzz.token_set_ratio(q_lower, content),
            fuzz.token_set_ratio(q_lower, summary) if summary else 0,
        )
        score += text_tsr * 0.60   # 100 → +60, 84 → +50, 0 → +0

        # Secondary text fields at lower weight
        subtasks_str   = " ".join(
            s.get("task", "").lower() for s in tags.get("subtasks", []) if isinstance(s, dict)
        )
        split_from_str = str(tags.get("split_from", "")).lower()
        original_dump  = str(tags.get("original_dump", "")).lower()
        if subtasks_str:
            score += fuzz.token_set_ratio(q_lower, subtasks_str) * 0.15   # max +15
        if split_from_str:
            score += fuzz.token_set_ratio(q_lower, split_from_str) * 0.12  # max +12
        if original_dump:
            score += fuzz.token_set_ratio(q_lower, original_dump) * 0.12   # max +12

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
    # Natural response (reserved for future product iterations)
    # Currently not called — search returns raw results only.
    # Restore the call in search() when an AI summary feature is added.
    # ──────────────────────────────────────────────────────────────

    # async def _natural_response(
    #     self, query: str, results: List[Dict],
    #     date_from: Optional[str], date_to: Optional[str], today: date,
    # ) -> str:
    #     if not results:
    #         time_ctx = ""
    #         if date_from:
    #             if date_from == date_to:
    #                 d        = datetime.fromisoformat(date_from).strftime("%A, %d %b")
    #                 time_ctx = f" for {d}"
    #             else:
    #                 time_ctx = f" between {date_from} and {date_to}"
    #         return f"Nothing found{time_ctx} in your notes."
    #
    #     saved_dates  = {r["_saved_date"] for r in results if r.get("_saved_date")}
    #     multi_source = len(saved_dates) > 1
    #     today_str    = str(today)
    #     yesterday    = str(today - timedelta(days=1))
    #
    #     if date_from and date_to:
    #         if date_from == date_to == today_str:    time_ctx = "today"
    #         elif date_from == date_to == yesterday:   time_ctx = "yesterday"
    #         elif date_from == date_to:                time_ctx = datetime.fromisoformat(date_from).strftime("%A, %d %b")
    #         else:                                     time_ctx = f"between {date_from} and {date_to}"
    #     else:
    #         time_ctx = "any time"
    #
    #     context = ""
    #     for r in results[:8]:
    #         due       = f" [due: {r['due_date']}]" if r.get("due_date") else ""
    #         ev_time   = f" at {r['event_time']}" if r.get("event_time") else ""
    #         buckets   = ", ".join(r.get("all_buckets", [r.get("category", "?")]))
    #         date_note = (
    #             f" [saved {_date_label(r.get('_saved_date', ''), today)}]"
    #             if multi_source else ""
    #         )
    #         context += f"\n[{buckets}]{due}{ev_time}{date_note}: {r['content']}\n"
    #
    #     prompt = f"""Personal assistant answering a search in the user's notes.
    #
    # TODAY: {today}
    # USER ASKED: "{query}"
    # TIME SCOPE: {time_ctx}
    #
    # MATCHING NOTES:
    # {context}
    #
    # Instructions:
    # - Answer the specific question directly
    # - List tasks/events clearly with times if available
    # - Be concise — max 4 sentences or a short list
    # - If results are from different dates, mention dates naturally (e.g. "yesterday you saved...")
    # - Never invent information not in the notes
    # - Don't say "based on your notes"
    #
    # Reply:"""
    #
    #     try:
    #         return await self.cerebras.chat_text(prompt, max_tokens=250, temperature=0.2)
    #     except Exception as e:
    #         print(f"[search] _natural_response failed ({type(e).__name__}): {e}")
    #         return f"Found {len(results)} result(s) matching your query."

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

    async def _fetch_images_direct(
        self,
        user_id: int,
        query: str,
        db: AsyncSession,
        limit: int = 5,
    ) -> Dict:
        """
        Search saved images by keyword match on content, summary, and tags.
        Returns file_id in results for Telegram re-send.
        Zero LLM — pure DB search.
        """
        # Extract search terms — strip "image:" prefix
        search_term = re.sub(r"^image:\s*", "", query, flags=re.IGNORECASE).strip().lower()

        if not search_term:
            return {"results": [], "natural_response": "", "is_image_search": True}

        # Split into individual words for broader matching
        terms = [t for t in search_term.split() if len(t) > 2]
        if not terms:
            terms = [search_term]

        # Build keyword conditions across content, summary, and extracted_text tag
        kw_conditions = []
        for idx, term in enumerate(terms[:6]):
            pat = f"%{term}%"
            kw_conditions.extend([
                func.lower(Message.content).contains(term),
                func.lower(Message.summary).contains(term),
                text(f"lower(messages.tags->>'extracted_text') LIKE :img_et_{idx}").bindparams(**{f"img_et_{idx}": pat}),
                text(f"lower(messages.tags->>'caption') LIKE :img_cap_{idx}").bindparams(**{f"img_cap_{idx}": pat}),
                text(f"lower(messages.tags->>'image_title') LIKE :img_title_{idx}").bindparams(**{f"img_title_{idx}": pat}),
            ])

        result = await db.execute(
            select(Message)
            .where(
                and_(
                    Message.user_id == user_id,
                    text("(messages.tags->>'is_image')::boolean IS TRUE"),
                    or_(*kw_conditions),
                )
            )
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        rows = result.scalars().all()

        if not rows:
            return {
                "results":          [],
                "natural_response": f"No images found matching *{search_term}*.",
                "is_image_search":  True,
            }

        results = []
        for msg in rows:
            tags = msg.tags if isinstance(msg.tags, dict) else {}
            results.append({
                "id":            msg.id,
                "content":       msg.content,
                "essence":       msg.summary or tags.get("image_title", "Image"),
                "message_type":  "image",
                "media_url":     msg.media_url,
                "file_id":       tags.get("file_id"),
                "caption":       tags.get("caption", ""),
                "document_type": tags.get("document_type", "other"),
                "description":   tags.get("description", ""),
                "created_at":    msg.created_at.isoformat(),
                "tags":          tags,
            })

        return {
            "results":         results,
            "natural_response": "",
            "is_image_search": True,
        }