import os
import httpx
from typing import List


GEMINI_EMBEDDING_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models"
    "/gemini-embedding-001:embedContent"
)


class EmbeddingService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        api_key: str | None = None,
        output_dimensionality: int | None = None,  # None = full 3072 dims
        task_type: str = "RETRIEVAL_DOCUMENT",      # see task types below
    ):
        # avoid re-init on singleton reuse
        if hasattr(self, "_initialized"):
            return
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        self._output_dimensionality = output_dimensionality
        self._task_type = task_type
        self._initialized = True

    # ─────────────────────────────────────────────────────────────
    # Public interface  (same as before)
    # ─────────────────────────────────────────────────────────────

    def embed(self, text: str) -> List[float]:
        """Embed a single string synchronously."""
        return self._embed_one(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple strings.
        Gemini embedding API only accepts one text per request,
        so this loops and calls _embed_one for each.
        """
        return [self._embed_one(t) for t in texts]

    # ─────────────────────────────────────────────────────────────
    # Core request
    # ─────────────────────────────────────────────────────────────

    def _embed_one(self, text: str) -> List[float]:
        payload = {
            "taskType": self._task_type,
            "content": {
                "parts": [{"text": text}]
            },
        }
        if self._output_dimensionality:
            payload["outputDimensionality"] = self._output_dimensionality

        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self._api_key,
        }

        response = httpx.post(GEMINI_EMBEDDING_URL, headers=headers, json=payload, timeout=15.0)
        response.raise_for_status()
        return response.json()["embedding"]["values"]


# Singleton instance — drop-in replacement for the old embedding_service
embedding_service = EmbeddingService(output_dimensionality=384)