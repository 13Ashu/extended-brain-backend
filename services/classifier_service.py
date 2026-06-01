"""
classifier_service.py — On-server intent bucket classifier (ONNX, ~10ms/msg)

Loads the fine-tuned all-MiniLM-L6-v2 backbone (ONNX) + sklearn logistic
regression head (numpy weights) trained in the intent-classifier-poc repo.

Replaces the Gemini LLM call for bucket classification. Gemini is still used
optionally for essence generation and entity extraction (non-blocking).

Usage:
    from services.classifier_service import classifier_service
    bucket, confidence = classifier_service.classify("call mom at 10pm")
    # → ("To-Do", 0.97)
"""

from __future__ import annotations

import os
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

LABELS   = ["Remember", "To-Do", "Ideas", "Track", "Events", "Random"]
# Low-confidence threshold — below this, fall back to Gemini
CONF_THRESHOLD = 0.50
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "intent_classifier")


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class ClassifierService:

    def __init__(self, model_dir: str = MODEL_DIR):
        self._model_dir  = os.path.abspath(model_dir)
        self._session    = None   # onnxruntime.InferenceSession
        self._tokenizer  = None   # transformers.AutoTokenizer
        self._W: Optional[np.ndarray] = None   # [n_classes, 384]
        self._b: Optional[np.ndarray] = None   # [n_classes]
        self._classes: Optional[np.ndarray] = None  # sorted label strings
        self._ready      = False

    def _maybe_download_backbone(self) -> None:
        """Download backbone.onnx from BACKBONE_ONNX_URL env var if the file is absent."""
        onnx_path = os.path.join(self._model_dir, "backbone.onnx")
        if os.path.exists(onnx_path):
            return
        url = os.environ.get("BACKBONE_ONNX_URL", "").strip()
        if not url:
            return

        import urllib.request
        import time

        logger.info(f"[classifier] backbone.onnx absent — downloading from BACKBONE_ONNX_URL (~86 MB, may take 10–30s) ...")
        os.makedirs(self._model_dir, exist_ok=True)
        tmp_path = onnx_path + ".tmp"
        t0 = time.time()

        def _progress(block_count, block_size, total_size):
            if total_size > 0 and block_count % 500 == 0:
                pct = min(100, int(block_count * block_size * 100 / total_size))
                logger.info(f"[classifier] downloading... {pct}%")

        try:
            urllib.request.urlretrieve(url, tmp_path, reporthook=_progress)
            os.rename(tmp_path, onnx_path)
            elapsed = time.time() - t0
            size_mb = os.path.getsize(onnx_path) // 1_000_000
            logger.info(f"[classifier] backbone.onnx downloaded ({size_mb} MB in {elapsed:.1f}s)")
        except Exception as e:
            logger.error(f"[classifier] backbone.onnx download failed: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def load(self) -> bool:
        """Load ONNX model + head weights. Called once at startup. Returns True on success."""
        self._maybe_download_backbone()
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer

            onnx_path = os.path.join(self._model_dir, "backbone.onnx")
            head_path = os.path.join(self._model_dir, "head_weights.npz")

            if not os.path.exists(onnx_path):
                logger.warning(f"[classifier] backbone.onnx not found at {onnx_path} — falling back to Gemini")
                return False
            if not os.path.exists(head_path):
                logger.warning(f"[classifier] head_weights.npz not found at {head_path} — falling back to Gemini")
                return False

            logger.info("[classifier] Loading ONNX backbone...")
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 2
            self._session = ort.InferenceSession(onnx_path, sess_opts)

            logger.info("[classifier] Loading tokenizer...")
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_dir)

            logger.info("[classifier] Loading head weights...")
            data           = np.load(head_path, allow_pickle=True)
            self._W        = data["W"].astype(np.float32)       # [n_classes, 384]
            self._b        = data["b"].astype(np.float32)       # [n_classes]
            self._classes  = data["classes"]                    # alphabetical order

            self._ready = True
            logger.info(f"[classifier] Ready — {len(self._classes)} classes: {list(self._classes)}")
            return True

        except ImportError as e:
            logger.warning(f"[classifier] Missing dependency ({e}) — falling back to Gemini")
            return False
        except Exception as e:
            logger.error(f"[classifier] Load failed: {e}")
            return False

    def classify(self, text: str) -> tuple[str, float]:
        """
        Classify text into one of 7 buckets.
        Returns (bucket_name, confidence) — confidence in [0, 1].
        Falls back to ("", 0.0) if model not loaded (caller uses Gemini).
        """
        if not self._ready:
            return "", 0.0

        try:
            enc = self._tokenizer(
                [text],
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=128,
            )
            # ONNX backbone → sentence embedding [1, 384]
            embedding = self._session.run(None, {
                "input_ids":      enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            })[0]  # [1, 384]

            # Logistic regression: logits = W @ embedding.T + b
            logits = (embedding @ self._W.T) + self._b  # [1, n_classes]
            probs  = _softmax(logits)[0]                # [n_classes]

            pred_idx   = int(probs.argmax())
            bucket     = str(self._classes[pred_idx])
            confidence = float(probs[pred_idx])
            return bucket, confidence

        except Exception as e:
            logger.error(f"[classifier] Inference error: {e}")
            return "", 0.0

    def classify_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        """Batch classify for throughput efficiency."""
        if not self._ready or not texts:
            return [("", 0.0)] * len(texts)

        try:
            enc = self._tokenizer(
                texts,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=128,
            )
            embeddings = self._session.run(None, {
                "input_ids":      enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            })[0]  # [batch, 384]

            logits = (embeddings @ self._W.T) + self._b  # [batch, n_classes]
            probs  = _softmax(logits)                    # [batch, n_classes]

            results = []
            for row in probs:
                idx = int(row.argmax())
                results.append((str(self._classes[idx]), float(row[idx])))
            return results

        except Exception as e:
            logger.error(f"[classifier] Batch inference error: {e}")
            return [("", 0.0)] * len(texts)

    @property
    def is_ready(self) -> bool:
        return self._ready


# Singleton — loaded once at app startup via classifier_service.load()
classifier_service = ClassifierService()
