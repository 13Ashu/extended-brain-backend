"""
AI Client for LLM operations
Supports Cerebras Cloud API and OpenRouter
Fast inference for categorization, tagging, and search
Updated for intelligent knowledge processing
"""

import os
import json
from typing import List, Dict, Optional, Any
import httpx


# ─────────────────────────────────────────────────────────────────────────────
# Provider config
# ─────────────────────────────────────────────────────────────────────────────

CEREBRAS_BASE_URL  = "https://api.cerebras.ai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

CEREBRAS_DEFAULT_MODEL    = "llama3.1-8b"
OPENROUTER_DEFAULT_MODEL  = "qwen/qwen3-vl-235b-a22b-thinking"   # fast + cheap default


class CerebrasClient:
    """
    Unified AI client supporting:
      - Cerebras Cloud  (provider="cerebras")
      - OpenRouter      (provider="openrouter")

    Both providers expose an OpenAI-compatible /chat/completions endpoint,
    so the core request logic is shared — only auth headers differ.

    Usage examples
    ──────────────
    # Cerebras (original behaviour, unchanged)
    client = CerebrasClient()

    # OpenRouter with default model
    client = CerebrasClient(provider="openrouter")

    # OpenRouter with a specific model
    client = CerebrasClient(
        provider="openrouter",
        model="anthropic/claude-3.5-sonnet",
        site_url="https://myapp.com",
        site_name="My App",
    )
    """

    def __init__(
        self,
        api_key:   str | None = None,
        provider:  str = "cerebras",          # "cerebras" | "openrouter"
        model:     str | None = None,
        # OpenRouter-specific optional metadata (used for rankings)
        site_url:  str | None = None,
        site_name: str | None = None,
    ):
        self.provider = provider.lower()

        if self.provider == "openrouter":
            self.api_key  = api_key or os.getenv("OPENROUTER_API_KEY")
            self.base_url = OPENROUTER_BASE_URL
            self.model    = model or OPENROUTER_DEFAULT_MODEL
            self.site_url  = site_url  or os.getenv("OPENROUTER_SITE_URL", "")
            self.site_name = site_name or os.getenv("OPENROUTER_SITE_NAME", "")
        else:
            # Default: Cerebras (original behaviour)
            self.api_key  = api_key or os.getenv("CEREBRAS_API_KEY")
            self.base_url = CEREBRAS_BASE_URL
            self.model    = model or CEREBRAS_DEFAULT_MODEL
            self.site_url  = ""
            self.site_name = ""

    # ─────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────

    async def chat(
        self,
        prompt: str,
        max_tokens: int = 800,
        temperature: float = 0.7,
        response_format: str = "json",
    ) -> Dict[str, Any]:
        """
        Main chat interface — returns parsed JSON.
        Used by the intelligent message processor and search.
        """

        if response_format == "json" and "Return JSON" not in prompt:
            prompt += "\n\nIMPORTANT: Return ONLY valid JSON, no markdown formatting, no extra text."

        response_text = await self._chat_completion(prompt, max_tokens, temperature)
        response_text = self._clean_json_response(response_text)

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Raw response: {response_text[:500]}")
            return {
                "error": "Failed to parse AI response",
                "raw":   response_text[:200],
            }

    # ─────────────────────────────────────────────────────────────
    # Core API call  (shared by both providers)
    # ─────────────────────────────────────────────────────────────

    async def _chat_completion(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        """
        Make a /chat/completions request.
        Both Cerebras and OpenRouter use the same OpenAI-compatible schema;
        only the base URL and auth headers differ.
        """

        headers = self._build_headers()

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a highly intelligent assistant. "
                        "Always follow instructions precisely. "
                        "When asked for JSON, return ONLY valid JSON "
                        "with no extra text or markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()

            except httpx.HTTPStatusError as e:
                print(f"[{self.provider}] HTTP error {e.response.status_code}: {e.response.text}")
                raise
            except Exception as e:
                print(f"[{self.provider}] API error: {e}")
                raise

    def _build_headers(self) -> Dict[str, str]:
        """Build provider-specific request headers."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if self.provider == "openrouter":
            # Optional but recommended for OpenRouter rankings
            if self.site_url:
                headers["HTTP-Referer"] = self.site_url
            if self.site_name:
                headers["X-OpenRouter-Title"] = self.site_name

        return headers

    # ─────────────────────────────────────────────────────────────
    # JSON cleaning
    # ─────────────────────────────────────────────────────────────

    def _clean_json_response(self, text: str) -> str:
        """Strip markdown fences and isolate the JSON object."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        start = text.find('{')
        end   = text.rfind('}')
        if start != -1 and end != -1:
            text = text[start:end + 1]

        return text.strip()

    # ─────────────────────────────────────────────────────────────
    # Legacy / convenience methods (unchanged API surface)
    # ─────────────────────────────────────────────────────────────

    async def categorize_message(
        self,
        content: str,
        existing_categories: List[str],
        message_type: str = "text",
    ) -> Dict[str, Any]:
        """Legacy method — kept for backward compatibility."""

        categories_text = ", ".join(existing_categories) if existing_categories else "None yet"

        prompt = f"""You are an intelligent assistant helping organize information.

Analyze this {message_type} message and provide:
1. Best category (choose from existing: {categories_text}, or suggest a new one)
2. Relevant tags (3-5 keywords)
3. Brief summary (1-2 sentences)
4. Extracted entities (people, dates, locations, etc.)

Message: {content}

Return JSON:
{{
    "category": "category name",
    "tags": ["tag1", "tag2", "tag3"],
    "summary": "brief summary here",
    "entities": {{
        "people": ["name1"],
        "dates": ["date1"],
        "locations": ["place1"],
        "organizations": ["org1"]
    }}
}}"""

        return await self.chat(prompt, response_format="json")

    async def search_query_enhancement(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Enhance search query with synonyms and related terms."""

        prompt = f"""Enhance this search query for better results.

Original query: {query}
{f"User context: {context}" if context else ""}

Provide:
1. Enhanced query with synonyms
2. Related search terms
3. Suggested filters

Return JSON:
{{
    "enhanced_query": "enhanced search terms",
    "synonyms": ["synonym1", "synonym2"],
    "related_terms": ["related1", "related2"],
    "suggested_filters": {{"category": ["cat1"], "time": "recent"}}
}}"""

        return await self.chat(prompt, response_format="json")

    async def answer_question(
        self,
        question: str,
        context_messages: List[Dict[str, str]],
    ) -> str:
        """Answer user's question based on their stored messages."""

        context = "\n\n".join([
            f"[{msg.get('category', 'N/A')}] {msg.get('content', '')}"
            for msg in context_messages
        ])

        prompt = f"""You are a helpful assistant with access to the user's personal knowledge base.

User's question: {question}

Relevant information from their notes:
{context}

Provide a helpful, concise answer based on the information above. If you can't answer from the given context, say so."""

        return await self._chat_completion(prompt, max_tokens=500)

    async def suggest_category_name(
        self,
        sample_messages: List[str],
        existing_categories: List[str],
    ) -> str:
        """Suggest a good category name for similar messages."""

        prompt = f"""Suggest a concise category name for these related messages:

Messages:
{chr(10).join(f"- {msg}" for msg in sample_messages)}

Existing categories: {", ".join(existing_categories)}

Respond with just the category name (2-3 words max)."""

        return await self._chat_completion(prompt, max_tokens=50)

    async def extract_document_text(self, text: str, max_length: int = 5000) -> str:
        """Summarize and extract key information from long documents."""

        if len(text) <= max_length:
            return text

        prompt = f"""Summarize this document, extracting the most important information:

{text[:10000]}

Provide a comprehensive summary that captures all key points, dates, names, and important details."""

        return await self._chat_completion(prompt, max_tokens=1000)

    async def transcribe_audio(self, audio_url: str) -> str:
        """Placeholder — integrate with Whisper / AssemblyAI / Google STT."""
        return f"[Audio transcription from {audio_url}]"

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Placeholder for vector embeddings.
        Neither Cerebras nor OpenRouter host dedicated embedding models today;
        use OpenAI /v1/embeddings or a local sentence-transformer instead.
        """
        return []
