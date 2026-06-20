"""Regression test for the citation hallucination-gate (_verify_citation).

Locks the case-normalization fix: a letter-suffix section the model cites (124A, 304B,
326A, 376D, …) must verify against the same retrieved section regardless of case, while
a genuinely absent section still trips the flag and generic/Article references still pass.

Usage:
    python scripts/verify_citation_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.rag.models import Chunk, RetrievedChunk  # noqa: E402
from app.services.rag_service import _verify_citation  # noqa: E402

_failures = 0


def check(name: str, ok: bool) -> None:
    global _failures
    if not ok:
        _failures += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


def rc(section: str | None = None, article: str | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(id="x", text="t", source_type="statute", act="IPC", section=section, article=article),
        score=1.0,
    )


def main() -> int:
    print("Citation verifier (case-normalization) test\n" + "=" * 56)
    check("letter-suffix verifies (124A vs 124A)", _verify_citation("IPC Section 124A", [rc("124A")]))
    check("cross-case verifies (124a vs 124A)", _verify_citation("IPC Section 124a", [rc("124A")]))
    check("another suffix (304B)", _verify_citation("BNS Section 304B", [rc("304B")]))
    check("plain number still verifies", _verify_citation("BNS Section 318", [rc("318")]))
    check("absent section still flagged", not _verify_citation("BNS Section 420", [rc("318")]))
    check("Article reference passes (lenient)", _verify_citation("Article 21", [rc(article="21")]))
    check("generic reference passes", _verify_citation("General Legal Guidance", []))
    print("=" * 56)
    print("ALL CHECKS PASSED" if _failures == 0 else f"{_failures} CHECK(S) FAILED")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
