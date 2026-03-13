"""
Intelligent Search Service — v2
────────────────────────────────────────────────────────────────────────────
KEY UPGRADES OVER v1
  • Cross-bucket search  — "show todos for tomorrow" finds items tagged as
    To-Do OR Events that have due_date/event_date matching tomorrow
  • tags->>'all_buckets' JSON array search (PostgreSQL @> operator)
  • Semantic search is the PRIMARY signal; keywords are secondary
  • Single LLM call for query expansion (no multi-turn overhead)
  • Priority-aware ranking
  • Entity-boosted scoring (person name in query → boost messages with that person)
  • Natural response generated with full context
────────────────────────────────────────────────────────────────────────────
Save as: services/search_service.py
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

from sqlalchemy import and_, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from cerebras_client import CerebrasClient
from database import Category, Message, User

BUCKET_NAMES = ["Remember", "To-Do", "Ideas", "Track", "Events", "Random"]

BUCKET_ALIASES: Dict[str, str] = {
    # To-Do
    "todo": "To-Do", "todos": "To-Do", "to-do": "To-Do", "task": "To-Do",
    "tasks": "To-Do", "action": "To-Do", "actions": "To-Do",
    "pending": "To-Do", "due": "To-Do",
    # Remember
    "remember": "Remember", "memory": "Remember", "recall": "Remember",
    "note": "Remember", "saved": "Remember", "where": "Remember",
    # Ideas
    "idea": "Ideas", "ideas": "Ideas", "concept": "Ideas", "thought": "Ideas",
    # Track
    "track": "Track", "log": "Track", "habit": "Track", "progress": "Track",
    "logged": "Track",
    # Events
    "event": "Events", "events": "Events", "appointment": "Events",
    "meeting": "Events", "plan": "Events", "schedule": "Events",
    "scheduled": "Events", "calendar": "Events",
}

# These phrases mean "search by WHEN it's due/scheduled", not when it was saved
DUE_DATE_SIGNALS = {
    "due", "todo", "to-do", "task", "pending", "scheduled for",
    "planned for", "need to do", "have to", "should do", "must do",
    "for today", "for tomorrow", "for monday", "for tuesday",
    "for wednesday", "for thursday", "for friday", "for saturday",
    "for sunday", "for this week", "for next week",
}


# ─────────────────────────────────────────────────────────────────────────────
# Temporal parsing (pure Python, zero tokens)
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
        if ref.month == 12:
            last = ref.replace(day=31)
        else:
            last = ref.replace(month=ref.month + 1, day=1) - timedelta(days=1)
        return str(first), str(last)
    if re.search(r"last\s+month", q):
        first_this = ref.replace(day=1)
        last_prev  = first_this - timedelta(days=1)
        return str(last_prev.replace(day=1)), str(last_prev)

    # "past/last N days"
    m = re.search(r"(?:past|last)\s+(\d+)\s+days?", q)
    if m:
        n = int(m.group(1))
        return str(ref - timedelta(days=n)), str(ref)

    # Day names: "for monday", "on friday"
    day_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for i, d in enumerate(day_names):
        if re.search(rf"\b{d}\b", q):
            delta = (i - ref.weekday()) % 7
            if delta == 0:
                delta = 7
            target = ref + timedelta(days=delta)
            return str(target), str(target)

    # Month names
    MONTHS = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
    for name, num in MONTHS.items():
        if re.search(rf"\b{name}\b", q):
            year = ref.year if num >= ref.month else ref.year + 1
            first = date(year, num, 1)
            last  = date(year, num + 1, 1) - timedelta(days=1) if num < 12 else date(year, 12, 31)
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


def _extract_person_name(query: str) -> Optional[str]:
    """Simple heuristic: words after 'about', 'with', 'from', 'call' that are capitalized."""
    patterns = [
        r"\b(?:about|with|from|call|called|contact)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
    ]
    for p in patterns:
        m = re.search(p, query)
        if m:
            name = m.group(1)
            # Filter common non-names
            if name.lower() not in {"show", "find", "get", "search", "my", "me", "the", "a", "an"}:
                return name
    return None


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

        # ── Zero-token pre-processing ─────────────────────────────
        date_from, date_to  = _resolve_date_range(query, today)
        bucket_hint         = _detect_bucket(query)
        use_due_filter      = _is_due_date_query(query) and date_from is not None
        person_hint         = _extract_person_name(query)

        # Default todo queries (no date specified) to today only
        q_lower = query.lower()
        is_future_query = any(kw in q_lower for kw in {"upcoming", "future", "all", "everything"})
        if not date_from and bucket_hint == "To-Do" and not is_future_query:
            date_from = str(today)
            date_to   = str(today)
            use_due_filter = True  # ← important: activate the due_date filter path

        # ── LLM query expansion ───────────────────────────────────
        expansion = await self._expand_query(
            query=query,
            user=user,
            db=db,
            date_from=date_from,
            date_to=date_to,
            bucket_hint=bucket_hint,
        )

        # Fast parser wins over LLM for dates (more reliable)
        if date_from:
            expansion["date_from"] = date_from
            expansion["date_to"]   = date_to
        if bucket_hint:
            expansion["bucket_filter"] = bucket_hint
        if person_hint:
            expansion.setdefault("entities", [])
            if person_hint not in expansion["entities"]:
                expansion["entities"].insert(0, person_hint)

        # ── DB retrieval ──────────────────────────────────────────
        messages = await self._retrieve(
            user=user,
            query=query,
            expansion=expansion,
            use_due_filter=use_due_filter,
            db=db,
            limit=limit * 3,
        )

        # ── Rank ──────────────────────────────────────────────────
        ranked = self._rank(
            messages=messages,
            query=query,
            expansion=expansion,
            use_due_filter=use_due_filter,
        )[:limit]

        # ── Natural response ──────────────────────────────────────
        natural = await self._natural_response(
            query=query,
            results=ranked,
            date_from=date_from,
            date_to=date_to,
            today=today,
        )

        return {"results": ranked, "natural_response": natural}

    # ──────────────────────────────────────────────────────────────
    # LLM query expansion
    # ──────────────────────────────────────────────────────────────

    async def _expand_query(
        self,
        query: str,
        user: User,
        db: AsyncSession,
        date_from: Optional[str],
        date_to: Optional[str],
        bucket_hint: Optional[str],
    ) -> Dict:
        today_str = datetime.utcnow().strftime("%Y-%m-%d (%A, %d %B %Y)")
        user_cats = await self._get_user_categories(user.id, db)

        prompt = f"""You are expanding a search query for a personal knowledge base.

USER: {user.name} ({user.occupation})
TODAY: {today_str}
CATEGORIES: {", ".join(user_cats)}
QUERY: "{query}"
PRE-PARSED: date_from={date_from}, date_to={date_to}, bucket_hint={bucket_hint}

CROSS-BUCKET RULE:
  If the user asks for "todos for tomorrow", they also want EVENTS for tomorrow
  (because a scheduled call or meeting IS a todo too).
  So expand bucket_filter to include related buckets when relevant:
    "todos" query   → search To-Do AND Events
    "events" query  → search Events AND To-Do (if action-like)
    "remember"      → search Remember only
    "ideas"         → search Ideas only
    "track/log"     → search Track only

Return ONLY this JSON:
{{
  "intent": "find_specific | browse_category | time_based | topic_explore",
  "core_concepts": ["main concepts to search for"],
  "keywords": ["expanded keywords + synonyms — be generous, 6-10 terms"],
  "entities": ["specific people/places/things mentioned"],
  "bucket_filter": "{bucket_hint or 'null'}",
  "extra_buckets": ["additional buckets to also search — e.g. Events when searching todos"],
  "search_focus": "content | summary | tags | all"
}}"""

        # response = await self.cerebras.chat(prompt)
        response = await self.cerebras.chat_lite(prompt, max_tokens=400)
        response.setdefault("core_concepts", [])
        response.setdefault("keywords", [])
        response.setdefault("entities", [])
        response.setdefault("intent", "find_specific")
        response.setdefault("search_focus", "all")
        response.setdefault("extra_buckets", [])
        return response

    # ──────────────────────────────────────────────────────────────
    # DB retrieval  — cross-bucket aware
    # ──────────────────────────────────────────────────────────────

    async def _retrieve(
        self,
        user: User,
        query: str,
        expansion: Dict,
        use_due_filter: bool,
        db: AsyncSession,
        limit: int,
    ) -> List[tuple]:

        # ── Semantic search ───────────────────────────────────────
        semantic_hits: Dict[int, float] = {}
        try:
            from services.embedding_service import embedding_service
            query_embedding = await embedding_service.aembed_query(query)
            embedding_str   = f"[{','.join(map(str, query_embedding))}]"
            sem_sql = text("""
                SELECT m.id,
                       1 - (m.embedding <=> :emb ::vector) AS similarity
                FROM   messages m
                WHERE  m.user_id = :uid
                  AND  m.embedding IS NOT NULL
                ORDER  BY m.embedding <=> :emb ::vector
                LIMIT  :lim
            """)
            sem_result = await db.execute(
                sem_sql, {"emb": embedding_str, "uid": user.id, "lim": limit}
            )
            semantic_hits = {row.id: float(row.similarity) for row in sem_result}
        except Exception as e:
            print(f"⚠ Semantic search failed: {e}")

        # ── Keyword retrieval ─────────────────────────────────────
        stmt = (
            select(Message, Category)
            .join(Category, Message.category_id == Category.id, isouter=True)
            .where(Message.user_id == user.id)
        )

        # Bucket filter — include primary + extra buckets
        bucket_filter = expansion.get("bucket_filter")
        extra_buckets = expansion.get("extra_buckets", [])
        if isinstance(extra_buckets, str):
            extra_buckets = [extra_buckets]
        all_target_buckets = list(dict.fromkeys(
            ([bucket_filter] if bucket_filter and bucket_filter in BUCKET_NAMES else [])
            + [b for b in extra_buckets if b in BUCKET_NAMES]
        ))

        if all_target_buckets:
            # Match if PRIMARY category OR any bucket in all_buckets JSON array
            bucket_conditions = []
            for b in all_target_buckets:
                bucket_conditions.append(Category.name == b)
                # PostgreSQL JSONB array contains check
                bucket_conditions.append(
                    text(f"messages.tags->'all_buckets' @> '\"{b}\"'::jsonb")
                )
            stmt = stmt.where(or_(*bucket_conditions))

        # Date filter
        date_from = expansion.get("date_from")
        date_to   = expansion.get("date_to")
        if date_from and date_to:
            if use_due_filter:
                # Search by due_date OR event date in events array
                stmt = stmt.where(
                    or_(
                        and_(
                            text("messages.tags->>'due_date' >= :df"),
                            text("messages.tags->>'due_date' <= :dt"),
                        ),
                        # Also match events within the range
                        text(
                            "EXISTS ("
                            "  SELECT 1 FROM jsonb_array_elements(messages.tags->'events') ev"
                            "  WHERE ev->>'date' >= :df AND ev->>'date' <= :dt"
                            ")"
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

        # Keyword conditions — search content, summary, and tags

        # Keyword conditions — clean single pass
        kw_conds = []
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

        stmt = stmt.order_by(Message.created_at.desc()).limit(limit)
        kw_result = await db.execute(stmt)
        kw_rows   = kw_result.all()

        # ── Merge ────────────────────────────────────────────────
        kw_ids  = {m.id for m, _ in kw_rows}
        all_ids = kw_ids | set(semantic_hits.keys())
        if not all_ids:
            return []

        final_stmt = (
            select(Message, Category)
            .join(Category, Message.category_id == Category.id, isouter=True)
            .where(Message.id.in_(all_ids))
        )

        # Re-apply critical filters to merged set
        if all_target_buckets:
            bucket_conditions = []
            for b in all_target_buckets:
                bucket_conditions.append(Category.name == b)
                bucket_conditions.append(
                    text(f"messages.tags->'all_buckets' @> '\"{b}\"'::jsonb")
                )
            final_stmt = final_stmt.where(or_(*bucket_conditions))

        if date_from and date_to and use_due_filter:
            final_stmt = final_stmt.where(
                or_(
                    and_(
                        text("messages.tags->>'due_date' >= :df"),
                        text("messages.tags->>'due_date' <= :dt"),
                    ),
                    text(
                        "EXISTS ("
                        "  SELECT 1 FROM jsonb_array_elements(messages.tags->'events') ev"
                        "  WHERE ev->>'date' >= :df AND ev->>'date' <= :dt"
                        ")"
                    ),
                )
            ).params(df=date_from, dt=date_to)

        final_result = await db.execute(final_stmt)
        rows = final_result.all()

        for message, _ in rows:
            message._semantic_score = semantic_hits.get(message.id, 0.0)

        return list(rows)

    # ──────────────────────────────────────────────────────────────
    # Ranking
    # ──────────────────────────────────────────────────────────────

    def _rank(
        self,
        messages: List[tuple],
        query: str,
        expansion: Dict,
        use_due_filter: bool,
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
            })
        scored.sort(key=lambda x: (x["relevance"], x["priority"] == "high"), reverse=True)
        return scored

    def _score(
        self,
        message: Message,
        category: Optional[Category],
        query: str,
        expansion: Dict,
        use_due_filter: bool,
    ) -> float:
        score   = 0.0
        content = message.content.lower()
        summary = (message.summary or "").lower()
        tags    = message.tags if isinstance(message.tags, dict) else {}
        q_lower = query.lower()

        # Semantic (strongest signal)
        score += getattr(message, "_semantic_score", 0.0) * 20.0

        # Exact phrase match
        if q_lower in content:
            score += 12.0

        # Core concepts
        for c in expansion.get("core_concepts", []):
            if c.lower() in content: score += 6.0
            if c.lower() in summary: score += 4.0

        # Keywords
        for kw in expansion.get("keywords", []):
            kl = kw.lower()
            if kl in content: score += 2.5
            if kl in summary: score += 1.5

        # Entity match (person name, place, etc.)
        for entity in expansion.get("entities", []):
            el = entity.lower()
            if el in content: score += 5.0
            # Also check inside JSONB entities
            all_entities_text = str(tags.get("entities", {})).lower()
            if el in all_entities_text: score += 3.0

        # Bucket match (primary or multi-bucket)
        all_msg_buckets = tags.get("all_buckets", [])
        bucket_filter = expansion.get("bucket_filter")
        extra_buckets = expansion.get("extra_buckets", [])
        matched_buckets = ([bucket_filter] if bucket_filter else []) + (extra_buckets or [])
        for b in matched_buckets:
            if b and (
                (category and category.name == b)
                or b in all_msg_buckets
            ):
                score += 8.0

        # Due date / event date in range
        if use_due_filter:
            date_from = expansion.get("date_from")
            date_to   = expansion.get("date_to")
            if date_from and date_to:
                due = tags.get("due_date")
                if due and date_from <= due <= date_to:
                    score += 15.0
                # Check events
                for ev in tags.get("events", []):
                    if isinstance(ev, dict):
                        ev_date = ev.get("date", "")
                        if ev_date and date_from <= ev_date <= date_to:
                            score += 15.0

        # Priority boost
        if tags.get("priority") == "high":
            score += 2.0

        # Recency
        age = (datetime.utcnow() - message.created_at).days
        if age < 1:   score += 2.0
        elif age < 7: score += 1.0

        return score

    # ──────────────────────────────────────────────────────────────
    # Natural response
    # ──────────────────────────────────────────────────────────────

    async def _natural_response(
        self,
        query: str,
        results: List[Dict],
        date_from: Optional[str],
        date_to: Optional[str],
        today: date,
    ) -> str:
        if not results:
            time_ctx = ""
            if date_from:
                if date_from == date_to:
                    d = datetime.fromisoformat(date_from).strftime("%A, %d %b")
                    time_ctx = f" for {d}"
                else:
                    time_ctx = f" between {date_from} and {date_to}"
            return f"Nothing found{time_ctx} in your notes."

        yesterday = str(today - timedelta(days=1))
        if date_from and date_to:
            if date_from == date_to == str(today):
                time_ctx = "today"
            elif date_from == date_to == yesterday:
                time_ctx = "yesterday"
            elif date_from == date_to:
                time_ctx = datetime.fromisoformat(date_from).strftime("%A, %d %b")
            else:
                time_ctx = f"between {date_from} and {date_to}"
        else:
            time_ctx = "any time"

        context = ""
        for r in results[:8]:
            due      = f" [due: {r['due_date']}]" if r.get("due_date") else ""
            ev_time  = f" at {r['event_time']}" if r.get("event_time") else ""
            buckets  = ", ".join(r.get("all_buckets", [r.get("category", "?")]))
            saved    = r["created_at"][:10]
            context += f"\n[{buckets}]{due}{ev_time} saved {saved}: {r['content']}\n"

        prompt = f"""Personal assistant responding to a search in the user's notes.

TODAY: {today}
USER ASKED: "{query}"
TIME SCOPE: {time_ctx}

MATCHING NOTES:
{context}

Instructions:
- Answer the specific question directly (e.g. "your todos for tomorrow are: ...")
- List tasks/events clearly with times if available
- Be concise — max 4 sentences or a short list
- If asking about a person (e.g. "Kailash"), highlight those results
- Don't say "based on your notes" — just answer as if you know
- Never invent information not in the notes

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
                if start > 0:         pre = "…" + pre
                if end < len(content): pre = pre + "…"
                return pre
        return content[:150] + "…"

    async def _get_user(self, phone: str, db: AsyncSession) -> Optional[User]:
        result = await db.execute(select(User).where(User.phone_number == phone))
        return result.scalar_one_or_none()

    async def _get_user_categories(self, user_id: int, db: AsyncSession) -> List[str]:
        result = await db.execute(select(Category.name).where(Category.user_id == user_id))
        return [row[0] for row in result.all()]
