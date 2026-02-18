from fastembed import TextEmbedding
from typing import List

class EmbeddingService:
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _load_model(self):
        if self._model is None:
            print("Loading embedding model...")
            self._model = TextEmbedding("BAAI/bge-small-en-v1.5")  # 130MB, no torch
            print("✓ Embedding model loaded")
        return self._model
    
    def embed(self, text: str) -> List[float]:
        model = self._load_model()
        text = text[:512]
        embeddings = list(model.embed([text]))
        return embeddings[0].tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        model = self._load_model()
        texts = [t[:512] for t in texts]
        embeddings = list(model.embed(texts))
        return [e.tolist() for e in embeddings]

embedding_service = EmbeddingService()