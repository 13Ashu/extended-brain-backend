"""
Vision Service
─────────────────────────────────────────────────────────────
Extracts text, type, and keywords from images using vision LLMs.
Currently supports: Gemini (Google AI Studio)
Swappable via VISION_PROVIDER env var.
"""

from __future__ import annotations

import base64
import os
from typing import Dict, Optional

import httpx


# ─────────────────────────────────────────────────────────────
# Provider: Gemini via Google AI Studio
# ─────────────────────────────────────────────────────────────

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

VISION_PROMPT = """You are a search indexer for a personal knowledge base. Analyze this image.

Your goal: extract every piece of information that would help the user find this image later by typing words into a search box.

Think from the user's recall perspective: "What word, name, number, place, brand, or phrase would I type to find this?"

Return ONLY this JSON, no markdown:
{
  "document_type": "id_card|receipt|bill|photo|screenshot|handwritten_note|business_card|certificate|menu|ticket|other",
  "title": "Short descriptive title (e.g. 'Aadhaar Card - Rahul Sharma', 'Zara Receipt ₹4544', 'Handwritten To-Do List')",
  "extracted_text": "ALL text visible in the image, verbatim and complete. This is the most important field — it powers keyword search. Include every word, number, date, and symbol exactly as written.",
  "recall_terms": "Space-separated search terms a user might type to find this image. Think broadly: document type synonyms, names, amounts, organizations, places, topics, colors, context. E.g. for an Aadhaar card: 'aadhaar aadhar identity id card uid government india biometric'",
  "description": "One sentence describing what this image shows — used for semantic search."
}"""


async def _analyze_with_gemini(
    image_data: bytes,
    mime_type: str,
    model: str = "gemini-2.5-flash-lite",
) -> Dict:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_STUDIO_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    url        = GEMINI_API_URL.format(model=model)
    image_b64  = base64.b64encode(image_data).decode("utf-8")

    payload = {
        "contents": [{
            "parts": [
                {"text": VISION_PROMPT},
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data":      image_b64,
                    }
                },
            ]
        }],
        "generationConfig": {
            "temperature":     0.1,
            "maxOutputTokens": 1000,
        },
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            url,
            json=payload,
            params={"key": api_key},
        )
        resp.raise_for_status()
        data = resp.json()

    # Extract text from Gemini response
    raw = data["candidates"][0]["content"]["parts"][0]["text"]

    # Strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    import json
    return json.loads(raw.strip())


# ─────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────

class VisionService:

    def __init__(self):
        self.provider = os.getenv("VISION_PROVIDER", "gemini")
        self.model    = os.getenv("VISION_MODEL", "gemini-2.5-flash-lite")

    async def analyze_image(
        self,
        image_data: bytes,
        mime_type: str = "image/jpeg",
    ) -> Dict:
        try:
            if self.provider == "gemini":
                result = await _analyze_with_gemini(image_data, mime_type, self.model)
            else:
                raise ValueError(f"Unknown vision provider: {self.provider}")

            result.setdefault("document_type", "other")
            result.setdefault("title", "Image")
            result.setdefault("extracted_text", "")
            result.setdefault("recall_terms", "")
            result.setdefault("description", "")
            return result

        except Exception as e:
            print(f"[vision] Analysis failed: {e}")
            return {
                "document_type":  "other",
                "title":          "Image",
                "extracted_text": "",
                "recall_terms":   "",
                "description":    "Image analysis failed",
            }


vision_service = VisionService()