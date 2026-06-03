"""
Thin async Redis cache using Upstash REST API.
All cache misses / errors are silent — callers always get None on miss.
"""
import os
import json
import hashlib
from typing import Any, Optional

import httpx

_URL   = os.getenv("UPSTASH_REDIS_REST_URL",   "")
_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")
_OK    = bool(_URL and _TOKEN)

_HEADERS = {"Authorization": f"Bearer {_TOKEN}"}


async def cache_get(key: str) -> Optional[Any]:
    if not _OK:
        return None
    try:
        async with httpx.AsyncClient(timeout=2.0) as c:
            r = await c.get(f"{_URL}/get/{key}", headers=_HEADERS)
            val = r.json().get("result")
            return json.loads(val) if val else None
    except Exception:
        return None


async def cache_set(key: str, value: Any, ex: int = 300) -> None:
    if not _OK:
        return
    try:
        async with httpx.AsyncClient(timeout=2.0) as c:
            # Pipeline: single round-trip for SET with EX
            await c.post(
                f"{_URL}/pipeline",
                headers=_HEADERS,
                json=[["SET", key, json.dumps(value, default=str), "EX", str(ex)]],
            )
    except Exception:
        pass


async def cache_del(key: str) -> None:
    if not _OK:
        return
    try:
        async with httpx.AsyncClient(timeout=2.0) as c:
            await c.get(f"{_URL}/del/{key}", headers=_HEADERS)
    except Exception:
        pass


# ── Key builders ───────────────────────────────────────────────────────────

def search_key(user_id: int, query: str) -> str:
    h = hashlib.sha256(query.lower().strip().encode()).hexdigest()[:12]
    return f"em:s:{user_id}:{h}"


def bootstrap_key(user_id: int, group_id: Optional[int]) -> str:
    return f"em:b:{user_id}:{group_id or 'p'}"


async def cache_del_user_searches(user_id: int) -> None:
    """Delete all search cache entries for a user (called after content edits)."""
    if not _OK:
        return
    try:
        pattern = f"em:s:{user_id}:*"
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.post(
                f"{_URL}/pipeline",
                headers=_HEADERS,
                json=[["KEYS", pattern]],
            )
            data = r.json()
            keys = data[0].get("result", []) if data else []
            if keys:
                await c.post(
                    f"{_URL}/pipeline",
                    headers=_HEADERS,
                    json=[["DEL"] + keys],
                )
    except Exception:
        pass
