"""
Embedding Service - runs locally on Railways
Uses sentence-transformers (free, no API needed)
Model: all-MiniLM-L6-v2 (384 dims, fast, good quality)
"""

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class EmbeddingService:
    _instance = None
    _model = None
    
    def __new__(cls):
        """Singleton - load model once, reuse forever"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _load_model(self):
        """Lazy load - only when first needed"""
        if self._model is None:
            print("Loading embedding model... (first time only)")
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ Embedding model loaded")
        return self._model
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        model = self._load_model()
        # Truncate to avoid issues with very long notes
        text = text[:512]
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts at once (more efficient)"""
        model = self._load_model()
        texts = [t[:512] for t in texts]
        embeddings = model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()


# Singleton instance
embedding_service = EmbeddingService()
