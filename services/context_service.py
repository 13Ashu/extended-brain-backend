"""
Context Service — Multi-turn search context via Upstash Redis
─────────────────────────────────────────────────────────────
Stores last search query per user with 30-minute TTL.
Used to enable follow-up queries without repeating context.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Dict
from upstash_redis.asyncio import Redis


def _get_redis() -> Redis:
    return Redis(
        url=os.getenv("UPSTASH_REDIS_REST_URL", ""),
        token=os.getenv("UPSTASH_REDIS_REST_TOKEN", ""),
    )


class ContextService:
    """Manages per-user search context for multi-turn conversations."""

    CONTEXT_TTL = 1800  # 30 minutes

    async def set_search_context(self, user_id: int, query: str, result_summary: str) -> None:
        """Store the last search query and a brief result summary."""
        try:
            redis = _get_redis()
            payload = json.dumps({
                "query":          query,
                "result_summary": result_summary[:500],  # cap size
            })
            await redis.set(f"ctx:search:{user_id}", payload, ex=self.CONTEXT_TTL)
        except Exception as e:
            print(f"[context] Failed to set search context: {e}")

    async def get_search_context(self, user_id: int) -> Optional[Dict]:
        """Retrieve last search context if still fresh."""
        try:
            redis = _get_redis()
            raw = await redis.get(f"ctx:search:{user_id}")
            if raw:
                return json.loads(raw)
        except Exception as e:
            print(f"[context] Failed to get search context: {e}")
        return None

    async def clear_search_context(self, user_id: int) -> None:
        try:
            redis = _get_redis()
            await redis.delete(f"ctx:search:{user_id}")
        except Exception as e:
            print(f"[context] Failed to clear search context: {e}")

    async def set_last_action(self, user_id: int, action: str, data: Dict) -> None:
        """Store the last action for undo support."""
        try:
            redis = _get_redis()
            payload = json.dumps({"action": action, "data": data})
            await redis.set(f"ctx:action:{user_id}", payload, ex=300)  # 5 min TTL
        except Exception as e:
            print(f"[context] Failed to set last action: {e}")

    async def get_last_action(self, user_id: int) -> Optional[Dict]:
        try:
            redis = _get_redis()
            raw = await redis.get(f"ctx:action:{user_id}")
            if raw:
                return json.loads(raw)
        except Exception as e:
            print(f"[context] Failed to get last action: {e}")
        return None

    async def set_pending_confirmation(self, user_id: int, confirmation_type: str, data: Dict) -> None:
        """
        Store a pending confirmation (e.g. project grouping, subtask assignment).
        User must reply yes/no to proceed.
        """
        try:
            redis = _get_redis()
            payload = json.dumps({"type": confirmation_type, "data": data})
            await redis.set(f"ctx:confirm:{user_id}", payload, ex=300)  # 5 min TTL
        except Exception as e:
            print(f"[context] Failed to set confirmation: {e}")

    async def get_pending_confirmation(self, user_id: int) -> Optional[Dict]:
        try:
            redis = _get_redis()
            raw = await redis.get(f"ctx:confirm:{user_id}")
            if raw:
                return json.loads(raw)
        except Exception as e:
            print(f"[context] Failed to get confirmation: {e}")
        return None

    async def clear_pending_confirmation(self, user_id: int) -> None:
        try:
            redis = _get_redis()
            await redis.delete(f"ctx:confirm:{user_id}")
        except Exception as e:
            print(f"[context] Failed to clear confirmation: {e}")


# Singleton
context_service = ContextService()
