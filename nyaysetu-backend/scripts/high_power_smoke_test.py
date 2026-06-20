"""Offline smoke test for high-power mode plumbing — no API key, no network, no anthropic pkg.

Verifies the opt-in Claude backend wiring without calling the cloud:
  - the provider factory defaults to local Ollama and switches to Claude only when asked,
  - ClaudeClient.generate_json extracts the JSON object from a Claude-shaped response
    (thinking block + text block), via an injected fake client,
  - a missing API key fails loudly (not silently),
  - the JSON extractor is robust to fences / surrounding prose.

Usage:
    python scripts/high_power_smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.services.llm_service import ClaudeClient, LLMError, OllamaClient, _extract_json, get_llm  # noqa: E402

_failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global _failures
    if not ok:
        _failures += 1
    line = f"  [{'PASS' if ok else 'FAIL'}] {name}"
    if detail:
        line += f"  - {detail}"
    print(line)


class _Block:
    def __init__(self, type_: str, **kw) -> None:
        self.type = type_
        for k, v in kw.items():
            setattr(self, k, v)


class _Resp:
    def __init__(self, content) -> None:
        self.content = content


class _FakeMessages:
    def __init__(self, content) -> None:
        self._content = content
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _Resp(self._content)


class _FakeAnthropic:
    def __init__(self, content) -> None:
        self.messages = _FakeMessages(content)


def main() -> int:
    print("High-power mode smoke test (offline)\n" + "=" * 64)

    # --- JSON extractor ---
    print("\nJSON extraction:")
    check("plain object", _extract_json('{"a": 1}') == '{"a": 1}')
    check("fenced object", _extract_json('```json\n{"a": 1}\n```') == '{"a": 1}')
    check("prose around object", _extract_json('Here: {"a": 1}. Done.') == '{"a": 1}')

    # --- Provider factory ---
    print("\nProvider factory:")
    original = settings.llm_provider
    try:
        settings.llm_provider = "ollama"
        check("defaults to Ollama (free/local)", isinstance(get_llm(), OllamaClient))
        settings.llm_provider = "claude"
        check("switches to Claude when configured", isinstance(get_llm(), ClaudeClient))
        settings.llm_provider = "CLAUDE  "  # case/space tolerant
        check("provider match is case/space tolerant", isinstance(get_llm(), ClaudeClient))
    finally:
        settings.llm_provider = original

    # --- ClaudeClient.generate_json with an injected fake client ---
    print("\nClaudeClient.generate_json (fake client):")
    fake_content = [
        _Block("thinking", thinking=""),  # display omitted => empty thinking text
        _Block("text", text='{"answer": "Death or life imprisonment", "law_reference": "BNS Section 103"}'),
    ]
    c = ClaudeClient(api_key="test-key", model="claude-opus-4-8")
    c._client = _FakeAnthropic(fake_content)  # bypass real SDK/network
    out = c.generate_json("dummy prompt")
    check("parses JSON from text block", out.get("law_reference") == "BNS Section 103", str(out))
    check("ignores empty thinking block", out.get("answer", "").startswith("Death"))
    check("sent adaptive thinking", c._client.messages.last_kwargs.get("thinking") == {"type": "adaptive"})
    check("used configured model", c._client.messages.last_kwargs.get("model") == "claude-opus-4-8")

    # --- Missing key fails loudly ---
    # Hermetic: clear any ambient key from .env so the empty-key path is actually exercised
    # (ClaudeClient(api_key="") otherwise falls back to settings.anthropic_api_key).
    print("\nMissing API key:")
    _saved_key = settings.anthropic_api_key
    settings.anthropic_api_key = ""
    try:
        ClaudeClient(api_key="").generate_json("x")
        check("raises LLMError without key", False)
    except LLMError as e:
        check("raises LLMError without key", "ANTHROPIC_API_KEY" in str(e))
    finally:
        settings.anthropic_api_key = _saved_key

    # --- Non-JSON response is caught ---
    print("\nMalformed response:")
    bad = ClaudeClient(api_key="k")
    bad._client = _FakeAnthropic([_Block("text", text="I cannot answer that.")])
    try:
        bad.generate_json("x")
        check("raises LLMError on non-JSON", False)
    except LLMError:
        check("raises LLMError on non-JSON", True)

    print("=" * 64)
    print("ALL CHECKS PASSED" if _failures == 0 else f"{_failures} CHECK(S) FAILED")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
