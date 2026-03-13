"""
AI Client — v2
────────────────────────────────────────────────────────────────────────────
Supports Cerebras Cloud, OpenRouter, and Google Gemini.
KEY CHANGES:
  • Better JSON extraction (handles nested/partial JSON)
  • Async embedding via EmbeddingService (not in this file — see embedding_service.py)
  • Temperature 0.1 for JSON calls (more reliable structured output)
  • Retry with backoff on 429/500
────────────────────────────────────────────────────────────────────────────
Save as: cerebras_client.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional
import hashlib
from functools import lru_cache

import httpx

CEREBRAS_BASE_URL   = "https://api.cerebras.ai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GEMINI_BASE_URL     = "https://generativelanguage.googleapis.com/v1beta/models"

CEREBRAS_DEFAULT_MODEL   = "gpt-oss-120b"
OPENROUTER_DEFAULT_MODEL = "google/gemma-3-4b-it:free"
GEMINI_DEFAULT_MODEL     = "gemini-2.0-flash-lite"   # faster than 2.5-flash-lite, better JSON
GEMINI_LITE_MODEL = "gemini-2.0-flash-lite"

_response_cache: dict = {}  # simple TTL-less cache for identical prompts


import time
_last_request_time: float = 0.0
_min_request_gap: float = 2.0  # safe for 30 RPM combined

class CerebrasClient:
    """
    Unified AI client: Cerebras / OpenRouter / Gemini.
    Default: Gemini (most reliable JSON output).
    """

    def __init__(
        self,
        api_key:   Optional[str] = None,
        provider:  str = "cerebras",
        model:     Optional[str] = None,
        site_url:  Optional[str] = None,
        site_name: Optional[str] = None,
    ):
        self.provider = provider.lower()

        if self.provider == "openrouter":
            self.api_key   = api_key or os.getenv("OPENROUTER_API_KEY")
            self.base_url  = OPENROUTER_BASE_URL
            self.model     = model or OPENROUTER_DEFAULT_MODEL
            self.site_url  = site_url  or os.getenv("OPENROUTER_SITE_URL", "")
            self.site_name = site_name or os.getenv("OPENROUTER_SITE_NAME", "")
        elif self.provider == "gemini":
            self.api_key  = api_key or os.getenv("GEMINI_API_KEY")
            self.base_url = GEMINI_BASE_URL
            self.model    = model or GEMINI_DEFAULT_MODEL
            self.site_url = self.site_name = ""
        else:
            self.api_key  = api_key or os.getenv("CEREBRAS_API_KEY")
            self.base_url = CEREBRAS_BASE_URL
            self.model    = model or CEREBRAS_DEFAULT_MODEL
            self.site_url = self.site_name = ""

    # ──────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────

    async def chat(
        self,
        prompt: str,
        max_tokens: int = 800,
        temperature: float = 0.1,   # low temp for deterministic JSON
        response_format: str = "json",
    ) -> Dict[str, Any]:
        

        # Cache identical prompts (helps when user resends same message)
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        if cache_key in _response_cache:
            return _response_cache[cache_key]

        if "Return ONLY" not in prompt and "Return ONLY this JSON" not in prompt:
            prompt += (
                "\n\nIMPORTANT: Return ONLY valid JSON. "
                "No markdown, no ``` fences, no extra text before or after the JSON object."
            )

        text = await self._chat_completion(prompt, max_tokens, temperature)
        text = self._clean_json(text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from messy response
            extracted = self._extract_json(text)
            if extracted:
                # Store result before returning
                _response_cache[cache_key] = extracted  # add this before every return
                if len(_response_cache) > 200:       # prevent unbounded growth
                    _response_cache.clear()

                return extracted
            print(f"[CerebrasClient] JSON parse failed. Raw: {text[:300]}")
            return {"error": "parse_failed", "raw": text[:300]}

    async def _chat_completion(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> str:
        if self.provider == "gemini":
            return await self._gemini(prompt, max_tokens, temperature)
        return await self._openai_compat(prompt, max_tokens, temperature)

    # ──────────────────────────────────────────────────────────────
    # Provider implementations
    # ──────────────────────────────────────────────────────────────

    async def _gemini(self, prompt: str, max_tokens: int, temperature: float, model_override: Optional[str] = None,) -> str:
        global _last_request_time

        model = model_override or self.model

        # Throttle — enforce minimum gap between requests
        now = time.monotonic()
        gap = now - _last_request_time
        if gap < _min_request_gap:
            await asyncio.sleep(_min_request_gap - gap)
        _last_request_time = time.monotonic()

        url = f"{self.base_url}/{model}:generateContent"
        headers = {"Content-Type": "application/json", "X-goog-api-key": self.api_key}
        payload = {
            "system_instruction": {
                "parts": [{"text": (
                    "You are a precise assistant. "
                    "When asked for JSON, output ONLY the JSON object — "
                    "no markdown, no explanation, no ``` fences."
                )}]
            },
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
                "responseMimeType": "application/json",
            },
        }

        delays = [2, 5, 15]  # more patient backoff
        for attempt in range(4):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(url, headers=headers, json=payload)
                    
                    if resp.status_code == 429:
                        # Respect Retry-After header if present
                        retry_after = int(resp.headers.get("Retry-After", delays[min(attempt, 2)]))
                        print(f"[gemini] 429 rate limit — waiting {retry_after}s (attempt {attempt+1})")
                        await asyncio.sleep(retry_after)
                        continue
                        
                    resp.raise_for_status()
                    return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                    
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (500, 503) and attempt < 3:
                    await asyncio.sleep(delays[min(attempt, 2)])
                    continue
                print(f"[gemini] HTTP {e.response.status_code}: {e.response.text[:200]}")
                raise
            except Exception as e:
                if attempt < 3:
                    await asyncio.sleep(delays[min(attempt, 2)])
                    continue
                raise

        return "{}"


    # Add a lite method
    async def chat_lite(
        self,
        prompt: str,
        max_tokens: int = 400,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """Use Flash-Lite for Gemini, or standard path for other providers."""
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        if cache_key in _response_cache:
            return _response_cache[cache_key]

        # Route based on provider
        if self.provider == "gemini":
            text = await self._gemini(prompt, max_tokens, temperature, model_override=GEMINI_LITE_MODEL)
        else:
            # Cerebras/OpenRouter — just use normal completion, no lite variant
            text = await self._openai_compat(prompt, max_tokens, temperature)

        text = self._clean_json(text)
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            result = self._extract_json(text) or {"error": "parse_failed"}

        _response_cache[cache_key] = result
        if len(_response_cache) > 200:
            _response_cache.clear()
        return result

    async def _openai_compat(self, prompt: str, max_tokens: int, temperature: float) -> str:
        headers = self._build_headers()
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a precise assistant. "
                        "When asked for JSON, return ONLY valid JSON with no extra text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    resp.raise_for_status()
                    return resp.json()["choices"][0]["message"]["content"].strip()
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (429, 500) and attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
                raise

        return "{}"

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    def _build_headers(self) -> Dict[str, str]:
        h = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if self.provider == "openrouter":
            if self.site_url:  h["HTTP-Referer"]       = self.site_url
            if self.site_name: h["X-OpenRouter-Title"] = self.site_name
        return h

    def _clean_json(self, text: str) -> str:
        text = text.strip()
        # Strip markdown fences
        for fence in ("```json", "```"):
            if text.startswith(fence):
                text = text[len(fence):]
                break
        if text.endswith("```"):
            text = text[:-3]
        # Extract first JSON object or array
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            s = text.find(start_char)
            e = text.rfind(end_char)
            if s != -1 and e != -1 and e > s:
                return text[s:e + 1].strip()
        return text.strip()

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Try harder to parse JSON from a messy response."""
        # Find any JSON-like block
        patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # nested one level
            r'\{.*\}',  # greedy
        ]
        for pattern in patterns:
            m = re.search(pattern, text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    continue
        return None

    # ──────────────────────────────────────────────────────────────
    # Legacy methods (unchanged for backward compatibility)
    # ──────────────────────────────────────────────────────────────

    async def chat_with_image(
        self,
        prompt: str,
        image_path: str,
        mime_type: str = "image/jpeg",
        max_tokens: int = 800,
        temperature: float = 0.7,
    ) -> str:
        if self.provider != "gemini":
            raise NotImplementedError("Image input is only supported for provider='gemini'")
        import base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        url = f"{self.base_url}/{self.model}:generateContent"
        headers = {"Content-Type": "application/json", "X-goog-api-key": self.api_key}
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": mime_type, "data": image_data}},
                ]
            }],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, json=payload, timeout=30.0)
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

    async def categorize_message(self, content, existing_categories, message_type="text"):
        categories_text = ", ".join(existing_categories) if existing_categories else "None yet"
        prompt = f"""Analyze this {message_type} message and categorize it.
Existing categories: {categories_text}
Message: {content}
Return JSON:
{{"category": "name", "tags": ["t1","t2"], "summary": "brief", "entities": {{"people":[],"dates":[],"locations":[]}}}}"""
        return await self.chat(prompt)

    async def transcribe_audio(self, audio_url: str) -> str:
        return f"[Audio transcription from {audio_url}]"

    async def generate_embedding(self, text: str) -> List[float]:
        from services.embedding_service import embedding_service
        return embedding_service.embed(text)
