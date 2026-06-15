"""Ollama HTTP client. Uses /api/generate with format=json for structured output."""
from __future__ import annotations

import json

import httpx

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class LLMError(RuntimeError):
    pass


class OllamaClient:
    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout_s: int | None = None,
    ) -> None:
        self._host = (host or settings.ollama_host).rstrip("/")
        self._model = model or settings.ollama_model
        self._timeout = timeout_s or settings.ollama_timeout_s

    def warmup(self) -> None:
        """Best-effort cold-start warmup: load model into RAM with a tiny prompt."""
        url = f"{self._host}/api/generate"
        try:
            with httpx.Client(timeout=self._timeout) as client:
                client.post(
                    url,
                    json={
                        "model": self._model,
                        "prompt": "ok",
                        "stream": False,
                        "options": {"num_predict": 1},
                    },
                )
        except httpx.HTTPError as e:
            logger.warning("Ollama warmup failed (will retry on first request): %s", e)

    def generate_json(self, prompt: str) -> dict:
        """Generate a response and parse it as JSON.

        Uses Ollama's `format: "json"` to constrain output to valid JSON.
        Raises LLMError on transport, timeout, or parse failures.
        """
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2},
        }
        url = f"{self._host}/api/generate"
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException as e:
            raise LLMError(f"Ollama timeout after {self._timeout}s") from e
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama request failed: {e}") from e

        raw = data.get("response", "")
        if not raw:
            raise LLMError("Ollama returned empty response")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning("Ollama returned non-JSON: %s", raw[:300])
            raise LLMError(f"Could not parse LLM JSON: {e}") from e


def _extract_json(text: str) -> str:
    """Pull the first complete JSON object out of a model response.

    Robust to a stray ```json fence or a sentence before/after the object — we take
    from the first ``{`` to the last ``}``.
    """
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


class ClaudeClient:
    """High-power LLM backend (Anthropic Claude) — same generate_json contract as
    OllamaClient, so the RAG/analyze/drafting services use it interchangeably.

    Opt-in: only constructed when ``settings.llm_provider == "claude"``. Keeps the
    product's free/local default (Ollama) intact. The ``anthropic`` SDK is imported
    lazily so the package isn't required unless high-power mode is actually enabled.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None, timeout_s: int | None = None) -> None:
        self._api_key = api_key or settings.anthropic_api_key
        self._model = model or settings.high_power_model
        self._timeout = timeout_s or settings.anthropic_timeout_s
        self._client = None  # lazy

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise LLMError("High-power mode is enabled but ANTHROPIC_API_KEY is not set.")
        try:
            import anthropic
        except ImportError as e:
            raise LLMError("High-power mode needs the 'anthropic' package (pip install anthropic).") from e
        self._client = anthropic.Anthropic(api_key=self._api_key, timeout=self._timeout)
        return self._client

    def warmup(self) -> None:
        # Cloud model — no local cold start to warm. Surface a misconfiguration at
        # startup rather than on the first user query.
        if not self._api_key:
            logger.warning("llm_provider=claude but ANTHROPIC_API_KEY is empty — /ask will fail until it's set.")

    def generate_json(self, prompt: str) -> dict:
        client = self._ensure_client()
        try:
            resp = client.messages.create(
                model=self._model,
                max_tokens=8000,
                system="You are a precise legal-information assistant for India. Return ONLY the JSON "
                       "object the user's instructions ask for — no markdown fences, no prose around it.",
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}],
            )
        except LLMError:
            raise
        except Exception as e:  # anthropic.APIError etc. — keep the route's friendly 502 path
            raise LLMError(f"Claude request failed: {e}") from e

        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text").strip()
        if not text:
            raise LLMError("Claude returned no text content")
        try:
            return json.loads(_extract_json(text))
        except json.JSONDecodeError as e:
            logger.warning("Claude returned non-JSON: %s", text[:300])
            raise LLMError(f"Could not parse Claude JSON: {e}") from e


def get_llm():
    """Return the configured LLM backend. Defaults to local Ollama (free); returns the
    Claude high-power backend only when ``llm_provider`` is explicitly set to 'claude'."""
    if settings.llm_provider.strip().lower() == "claude":
        logger.info("LLM provider: Claude high-power mode (%s)", settings.high_power_model)
        return ClaudeClient()
    return OllamaClient()
