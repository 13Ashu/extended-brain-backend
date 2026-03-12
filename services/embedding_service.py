"""
Embedding Service — v2
────────────────────────────────────────────────────────────────────────────
KEY UPGRADES OVER v1
  • Full 3072-dim Gemini embeddings (was 384) — dramatically richer semantic space
  • Async-native: embed() and embed_batch() are both async
  • Sync wrapper embed_sync() kept for non-async callers (uses httpx.post directly)
  • Task-type switching: RETRIEVAL_DOCUMENT for storing, RETRIEVAL_QUERY for searching
  • Retry logic with exponential backoff
  • Singleton with lazy initialization
────────────────────────────────────────────────────────────────────────────
Save as: services/embedding_service.py

IMPORTANT: After upgrading dimensions, you MUST re-create the pgvector column:
  ALTER TABLE messages DROP COLUMN embedding;
  ALTER TABLE messages ADD COLUMN embedding vector(3072);
  CREATE INDEX ON messages USING hnsw (embedding vector_cosine_ops);
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import List, Optional

import httpx

GEMINI_EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models"
    "/gemini-embedding-001:embedContent"
)

# Full 3072 dims for maximum semantic richness
# Set to a smaller value (e.g. 768) if you hit pgvector index size limits
DEFAULT_DIMS = 1536


class EmbeddingService:
    _instance: Optional["EmbeddingService"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        api_key: Optional[str] = None,
        output_dimensionality: int = DEFAULT_DIMS,
    ):
        if hasattr(self, "_initialized"):
            return
        self._api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self._dims    = output_dimensionality
        self._initialized = True

    # ──────────────────────────────────────────────────────────────
    # Async interface (preferred)
    # ──────────────────────────────────────────────────────────────

    async def aembed(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
        """Async embed a single string."""
        return await self._async_embed_one(text, task_type)

    async def aembed_query(self, text: str) -> List[float]:
        """Async embed a search query (uses RETRIEVAL_QUERY task type for better retrieval)."""
        return await self._async_embed_one(text, "RETRIEVAL_QUERY")

    async def aembed_batch(
        self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> List[List[float]]:
        """Async embed multiple strings concurrently (up to 5 at a time)."""
        semaphore = asyncio.Semaphore(5)

        async def _bounded(t: str) -> List[float]:
            async with semaphore:
                return await self._async_embed_one(t, task_type)

        return await asyncio.gather(*[_bounded(t) for t in texts])

    # ──────────────────────────────────────────────────────────────
    # Sync interface (kept for backward compatibility)
    # ──────────────────────────────────────────────────────────────

    def embed(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
        """Synchronous embed — uses httpx.post directly."""
        return self._sync_embed_one(text, task_type)

    def embed_query(self, text: str) -> List[float]:
        return self._sync_embed_one(text, "RETRIEVAL_QUERY")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._sync_embed_one(t) for t in texts]

    # ──────────────────────────────────────────────────────────────
    # Core async request
    # ──────────────────────────────────────────────────────────────

    async def _async_embed_one(self, text: str, task_type: str) -> List[float]:
        payload = {
            "taskType": task_type,
            "content": {"parts": [{"text": text[:8000]}]},  # Gemini limit
        }
        if self._dims != 3072:
            payload["outputDimensionality"] = self._dims

        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self._api_key,
        }

        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=20.0) as client:
                    resp = await client.post(GEMINI_EMBED_URL, headers=headers, json=payload)
                    resp.raise_for_status()
                    return resp.json()["embedding"]["values"]
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                print(f"[embedding] HTTP {e.response.status_code}: {e.response.text[:200]}")
                raise
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
                print(f"[embedding] Error: {e}")
                raise

        return [0.0] * self._dims

    # ──────────────────────────────────────────────────────────────
    # Core sync request
    # ──────────────────────────────────────────────────────────────

    def _sync_embed_one(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
        payload = {
            "taskType": task_type,
            "content": {"parts": [{"text": text[:8000]}]},
        }
        if self._dims != 3072:
            payload["outputDimensionality"] = self._dims

        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self._api_key,
        }

        for attempt in range(3):
            try:
                resp = httpx.post(GEMINI_EMBED_URL, headers=headers, json=payload, timeout=20.0)
                resp.raise_for_status()
                return resp.json()["embedding"]["values"]
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise
            except Exception as e:
                if attempt < 2:
                    time.sleep(1)
                    continue
                raise

        return [0.0] * self._dims


# Singleton — drop-in replacement
embedding_service = EmbeddingService(output_dimensionality=DEFAULT_DIMS)
