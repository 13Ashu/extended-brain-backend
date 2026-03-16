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

VISION_PROMPT = """Analyze this image and extract the following in JSON format.
Be thorough with OCR — extract ALL visible text exactly as written.

Return ONLY this JSON, no markdown:
{
  "document_type": "id_card|receipt|handwritten_note|screenshot|photo|business_card|certificate|other",
  "title": "short descriptive title (e.g. 'Aadhaar Card - Rahul Sharma')",
  "extracted_text": "ALL text visible in the image, verbatim",
  "key_fields": {
    "name": "person name if visible",
    "number": "ID/receipt/reference number if visible",
    "date": "any date visible",
    "organization": "issuing org or brand",
    "address": "address if visible"
  },
  "description": "one sentence describing what this image shows",
  "keywords": ["5-10 searchable keywords"],
  "language": "english|hindi|mixed|other"
}"""


async def _analyze_with_gemini(
    image_data: bytes,
    mime_type: str,
    model: str = "gemini-2.0-flash-lite",
) -> Dict:
    api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY", "")
    if not api_key:
        raise ValueError("GOOGLE_AI_STUDIO_API_KEY not set")

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

# Remove download_telegram_image from VisionService entirely.
# The caller (main.py) handles the download using the existing messaging_client.

class VisionService:

    def __init__(self):
        self.provider = os.getenv("VISION_PROVIDER", "gemini")
        self.model    = os.getenv("VISION_MODEL", "gemini-2.0-flash-lite")

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
            result.setdefault("key_fields", {})
            result.setdefault("description", "")
            result.setdefault("keywords", [])
            result.setdefault("language", "english")
            return result

        except Exception as e:
            print(f"[vision] Analysis failed: {e}")
            return {
                "document_type":  "other",
                "title":          "Image",
                "extracted_text": "",
                "key_fields":     {},
                "description":    "Image analysis failed",
                "keywords":       [],
                "language":       "english",
            }


vision_service = VisionService()