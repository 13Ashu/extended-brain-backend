"""
Intelligent Search Service
Natural language search that understands INTENT, TIME, and BUCKET context.
"""

from typing import List, Dict, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, func, text
from datetime import datetime, timedelta, date
import re

from database import User, Message, Category
from cerebras_client import CerebrasClient
from services.embedding_service import embedding_service


# ─────────────────────────────────────────────────────────────────────────────
# Intent bucket names (must match message_processor.py)
# ─────────────────────────────────────────────────────────────────────────────
BUCKET_NAMES = ["Remember", "To-Do", "Ideas", "Track", "Events", "Random"]

BUCKET_ALIASES = {
    # To-Do
    "todo": "To-Do", "todos": "To-Do", "to-do": "To-Do", "task": "To-Do",
    "tasks": "To-Do", "action": "To-Do", "actions": "To-Do", "pending": "To-Do",
    # Remember
    "remember": "Remember", "memory": "Remember", "recall": "Remember",
    "note": "Remember", "saved": "Remember",
    # Ideas
    "idea": "Ideas", "ideas": "Ideas", "concept": "Ideas", "thought": "Ideas",
    # Track
    "track": "Track", "log": "Track", "habit": "Track", "progress": "Track",
    # Events
    "event": "Events", "events": "Events", "appointment": "Events",
    "meeting": "Events", "plan": "Events", "schedule": "Events",
}


# ─────────────────────────────────────────────────────────────────────────────
# Temporal parsing  (pure Python, no LLM tokens wasted)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_date_range(query: str, ref: date) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse natural-language time expressions and return (date_from, date_to) as
    YYYY-MM-DD strings, or (None, None) if no temporal signal found.
    """
    q = query.lower()

    if "yesterday" in q:
        d = ref - timedelta(days=1)
        return str(d), str(d)

    if "today" in q or "right now" in q:
        return str(ref), str(ref)

    if "tomorrow" in q:
        d = ref + timedelta(days=1)
        return str(d), str(d)

    if re.search(r"this\s+week", q):
        monday = ref - timedelta(days=ref.weekday())
        sunday = monday + timedelta(days=6)
        return str(monday), str(sunday)

    if re.search(r"last\s+week", q):
        monday = ref - timedelta(days=ref.weekday() + 7)
        sunday = monday + timedelta(days=6)
        return str(monday), str(sunday)

    if re.search(r"this\s+month", q):
        first = ref.replace(day=1)
        # last day of month
        if ref.month == 12:
            last = ref.replace(day=31)
        else:
            last = ref.replace(month=ref.month + 1, day=1) - timedelta(days=1)
        return str(first), str(last)

    if re.search(r"last\s+month", q):
        first_this = ref.replace(day=1)
        last_prev  = first_this - timedelta(days=1)
        first_prev = last_prev.replace(day=1)
        return str(first_prev), str(last_prev)

    # "past N days / last N days"
    m = re.search(r"(?:past|last)\s+(\d+)\s+days?", q)
    if m:
        n = int(m.group(1))
        return str(ref - timedelta(days=n)), str(ref)

    # Explicit month names: "in March", "march todos"
    MONTHS = {
        "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
        "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
    }
    for name, num in MONTHS.items():
        if name in q:
            year = ref.year if num <= ref.month else ref.year - 1
            first = date(year, num, 1)
            if num == 12:
                last = date(year, 12, 31)
            else:
                last = date(year, num + 1, 1) - timedelta(days=1)
            return str(first), str(last)

    return None, None


def _detect_bucket_from_query(query: str) -> Optional[str]:
    """Detect if the user is explicitly asking about a bucket type."""
    q = query.lower()
    for alias, bucket in BUCKET_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", q):
            return bucket
    return None


def _is_due_date_query(query: str) -> bool:
    """True if the temporal reference is about WHEN something is DUE, not when it was saved."""
    due_signals = ["due", "todo", "to-do", "task", "pending", "scheduled for", "planned for",
                   "need to do", "have to", "should do", "must do"]
    q = query.lower()
    return any(s in q for s in due_signals)


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

        # ── Fast pre-processing (no LLM) ──────────────────────────────────
        date_from, date_to   = _resolve_date_range(query, today)
        bucket_hint          = _detect_bucket_from_query(query)
        use_due_date_filter  = _is_due_date_query(query) and date_from is not None

        # ── Step 1: LLM search understanding ─────────────────────────────
        understanding = await self._understand_search_query(
            query=query,
            user=user,
            db=db,
            date_from=date_from,
            date_to=date_to,
            bucket_hint=bucket_hint,
        )

        # Merge fast-parsed date with LLM result (fast parser wins — it's more reliable)
        if date_from:
            understanding["date_from"] = date_from
            understanding["date_to"]   = date_to
        if bucket_hint:
            understanding["bucket_filter"] = bucket_hint

        # ── Step 2: DB search ─────────────────────────────────────────────
        messages = await self._intelligent_search(
            user=user,
            query=query,
            understanding=understanding,
            use_due_date_filter=use_due_date_filter,
            db=db,
            limit=limit * 3,
        )

        # ── Step 3: Rank ─────────────────────────────────────────────────
        ranked = await self._rank_by_relevance(
            messages=messages,
            query=query,
            understanding=understanding,
            use_due_date_filter=use_due_date_filter,
        )
        ranked = ranked[:limit]

        # ── Step 4: Natural response ──────────────────────────────────────
        natural_response = await self._generate_natural_response(
            query=query,
            results=ranked,
            date_from=date_from,
            date_to=date_to,
            today=today,
        )

        return {"results": ranked, "natural_response": natural_response}

    # ─────────────────────────────────────────────────────────────────────
    # Step 1 — LLM understanding
    # ─────────────────────────────────────────────────────────────────────

    async def _understand_search_query(
        self,
        query: str,
        user: User,
        db: AsyncSession,
        date_from: Optional[str],
        date_to: Optional[str],
        bucket_hint: Optional[str],
    ) -> Dict:
        user_categories = await self._get_user_categories(user.id, db)
        today_str = datetime.utcnow().strftime("%Y-%m-%d (%A, %d %B %Y)")

        prompt = f"""You are helping search a personal knowledge base.

USER: {user.name} ({user.occupation})
TODAY: {today_str}
CATEGORIES: {', '.join(user_categories)}
SEARCH QUERY: "{query}"
PRE-PARSED: date_from={date_from}, date_to={date_to}, bucket={bucket_hint}

Return ONLY this JSON (no markdown):
{{
  "intent": "find_specific | browse_category | time_based | topic_explore",
  "core_concepts": ["main concepts"],
  "keywords": ["expanded keywords + synonyms"],
  "entities": ["specific people/places/things"],
  "bucket_filter": "{bucket_hint or 'null — one of: Remember, To-Do, Ideas, Track, Events, Random, or null'}",
  "search_focus": "content | summary | tags | all"
}}"""

        response = await self.cerebras.chat(prompt)
        response.setdefault("core_concepts", [])
        response.setdefault("keywords", [])
        response.setdefault("entities", [])
        response.setdefault("intent", "find_specific")
        response.setdefault("search_focus", "all")
        return response

    # ─────────────────────────────────────────────────────────────────────
    # Step 2 — DB search
    # ─────────────────────────────────────────────────────────────────────

    async def _intelligent_search(
        self,
        user: User,
        query: str,
        understanding: Dict,
        use_due_date_filter: bool,
        db: AsyncSession,
        limit: int,
    ) -> List[tuple]:

        # ── Semantic search ───────────────────────────────────────────────
        semantic_hits: Dict[int, float] = {}
        try:
            query_embedding = embedding_service.embed(query)
            embedding_str   = f"[{','.join(map(str, query_embedding))}]"
            sem_result = await db.execute(text("""
                SELECT m.id, 1 - (m.embedding <=> :emb ::vector) AS similarity
                FROM messages m
                WHERE m.user_id = :uid AND m.embedding IS NOT NULL
                ORDER BY m.embedding <=> :emb ::vector
                LIMIT :lim
            """), {"emb": embedding_str, "uid": user.id, "lim": limit})
            semantic_hits = {row.id: row.similarity for row in sem_result}
        except Exception as e:
            print(f"⚠ Semantic search failed: {e}")

        # ── Keyword search ────────────────────────────────────────────────
        stmt = (
            select(Message, Category)
            .join(Category, Message.category_id == Category.id, isouter=True)
            .where(Message.user_id == user.id)
        )

        # Bucket filter (category name = bucket name)
        bucket_filter = understanding.get("bucket_filter")
        if bucket_filter and bucket_filter in BUCKET_NAMES:
            stmt = stmt.where(Category.name == bucket_filter)

        # Date filter — due_date (for to-do queries) OR created_at (for general)
        date_from = understanding.get("date_from")
        date_to   = understanding.get("date_to")
        if date_from and date_to:
            if use_due_date_filter:
                # Filter by tags->>'due_date' (JSONB text field)
                stmt = stmt.where(
                    and_(
                        text("messages.tags->>'due_date' >= :df"),
                        text("messages.tags->>'due_date' <= :dt"),
                    )
                ).params(df=date_from, dt=date_to)
            else:
                # Filter by when the message was saved
                stmt = stmt.where(
                    and_(
                        Message.created_at >= datetime.fromisoformat(date_from),
                        Message.created_at < datetime.fromisoformat(date_to) + timedelta(days=1),
                    )
                )

        # Keyword conditions
        kw_conditions = []
        all_terms = (
            understanding.get("keywords", [])[:6]
            + understanding.get("core_concepts", [])[:3]
            + understanding.get("entities", [])[:3]
        )
        for term in all_terms:
            kw_conditions.append(func.lower(Message.content).contains(term.lower()))
            kw_conditions.append(func.lower(Message.summary).contains(term.lower()))
        if kw_conditions:
            stmt = stmt.where(or_(*kw_conditions))

        stmt = stmt.order_by(Message.created_at.desc()).limit(limit)
        kw_result = await db.execute(stmt)
        kw_rows   = kw_result.all()

        # ── Merge semantic + keyword results ──────────────────────────────
        kw_ids  = {m.id for m, _ in kw_rows}
        all_ids = kw_ids | set(semantic_hits.keys())
        if not all_ids:
            return []

        final_stmt = (
            select(Message, Category)
            .join(Category, Message.category_id == Category.id, isouter=True)
            .where(Message.id.in_(all_ids))
        )

        # Re-apply bucket + date filters to semantic hits too
        if bucket_filter and bucket_filter in BUCKET_NAMES:
            final_stmt = final_stmt.where(Category.name == bucket_filter)
        if date_from and date_to:
            if use_due_date_filter:
                final_stmt = final_stmt.where(
                    and_(
                        text("messages.tags->>'due_date' >= :df"),
                        text("messages.tags->>'due_date' <= :dt"),
                    )
                ).params(df=date_from, dt=date_to)
            else:
                final_stmt = final_stmt.where(
                    and_(
                        Message.created_at >= datetime.fromisoformat(date_from),
                        Message.created_at < datetime.fromisoformat(date_to) + timedelta(days=1),
                    )
                )

        final_result = await db.execute(final_stmt)
        rows = final_result.all()

        for message, _ in rows:
            message._semantic_score = semantic_hits.get(message.id, 0.0)

        return [(m, c) for m, c in rows]

    # ─────────────────────────────────────────────────────────────────────
    # Step 3 — Rank
    # ─────────────────────────────────────────────────────────────────────

    async def _rank_by_relevance(
        self,
        messages: List[tuple],
        query: str,
        understanding: Dict,
        use_due_date_filter: bool,
    ) -> List[Dict]:
        scored = []
        for message, category in messages:
            score = self._score(message, category, query, understanding, use_due_date_filter)
            tags  = message.tags if isinstance(message.tags, dict) else {}
            scored.append({
                "id":         message.id,
                "content":    message.content,
                "essence":    message.summary,
                "category":   category.name if category else "Uncategorized",
                "tags":       tags,
                "created_at": message.created_at.isoformat(),
                "due_date":   tags.get("due_date"),
                "relevance":  score,
                "preview":    self._preview(message.content, understanding.get("keywords", [])),
            })
        scored.sort(key=lambda x: x["relevance"], reverse=True)
        return scored

    def _score(
        self,
        message: Message,
        category: Optional[Category],
        query: str,
        understanding: Dict,
        use_due_date_filter: bool,
    ) -> float:
        score   = 0.0
        content = message.content.lower()
        summary = (message.summary or "").lower()
        tags    = message.tags if isinstance(message.tags, dict) else {}

        # Semantic similarity (strongest signal)
        score += getattr(message, "_semantic_score", 0.0) * 15.0

        # Exact phrase
        if query.lower() in content:
            score += 10.0

        # Core concepts
        for concept in understanding.get("core_concepts", []):
            if concept.lower() in content: score += 5.0
            if concept.lower() in summary: score += 3.0

        # Keywords
        for kw in understanding.get("keywords", []):
            if kw.lower() in content: score += 2.0
            if kw.lower() in summary: score += 1.0

        # Entities
        for entity in understanding.get("entities", []):
            if entity.lower() in content: score += 3.0

        # Bucket match
        bucket_filter = understanding.get("bucket_filter")
        if bucket_filter and category and category.name == bucket_filter:
            score += 8.0

        # Due date proximity boost (for todo queries with date)
        if use_due_date_filter:
            due = tags.get("due_date")
            date_from = understanding.get("date_from")
            date_to   = understanding.get("date_to")
            if due and date_from and date_to and date_from <= due <= date_to:
                score += 12.0  # Strong boost — exactly what was asked for

        # Recency boost (softer — date filter already handles time scoping)
        age = (datetime.utcnow() - message.created_at).days
        if age < 1:   score += 2.0
        elif age < 7: score += 1.0

        return score

    # ─────────────────────────────────────────────────────────────────────
    # Step 4 — Natural response
    # ─────────────────────────────────────────────────────────────────────

    async def _generate_natural_response(
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

        # Build time context string for the LLM
        time_ctx = ""
        if date_from and date_to:
            yesterday = str(today - timedelta(days=1))
            if date_from == date_to == str(today):
                time_ctx = "today"
            elif date_from == date_to == yesterday:
                time_ctx = "yesterday"
            elif date_from == date_to:
                time_ctx = datetime.fromisoformat(date_from).strftime("%A, %d %b")
            else:
                time_ctx = f"between {date_from} and {date_to}"

        context = ""
        for r in results[:6]:
            due   = f" [due: {r['due_date']}]" if r.get("due_date") else ""
            saved = r["created_at"][:10]
            context += f"\n[{r['category']}]{due} saved {saved}: {r['content']}\n"

        prompt = f"""You are a personal assistant. The user searched their notes.

TODAY: {today}
USER ASKED: "{query}"
TIME SCOPE: {time_ctx or 'any time'}

MATCHING NOTES:
{context}

Answer naturally and directly. Rules:
- Answer the specific question (e.g. "your todos for yesterday were: ...")
- Be concise — max 3-4 sentences
- If it's a to-do query, list the tasks clearly
- Don't say "based on your notes" — just answer
- If there are dates, mention them naturally
- Never make up information not in the notes

Reply:"""

        return await self.cerebras._chat_completion(prompt, max_tokens=250, temperature=0.3)

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

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
                if start > 0: pre = "…" + pre
                if end < len(content): pre = pre + "…"
                return pre
        return content[:150] + "…"

    async def _get_user(self, phone: str, db: AsyncSession) -> Optional[User]:
        result = await db.execute(select(User).where(User.phone_number == phone))
        return result.scalar_one_or_none()

    async def _get_user_categories(self, user_id: int, db: AsyncSession) -> List[str]:
        result = await db.execute(select(Category.name).where(Category.user_id == user_id))
        return [row[0] for row in result.all()]
