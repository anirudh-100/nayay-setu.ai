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
