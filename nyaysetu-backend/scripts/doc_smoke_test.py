"""Offline smoke test for document understanding (/analyze) — no downloads, no Ollama.

Reuses the main smoke test's fakes (deterministic hashing embedder + temp index) and
substitutes a canned analyze-shaped LLM, so it proves the document-analysis
orchestration — retrieval, grounding, citation verification, current-law note, and the
hallucination gate — is wired correctly without any heavy externals.

Usage:
    python scripts/doc_smoke_test.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.smoke_test import FakeEmbedder, setup_index  # noqa: E402

_failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global _failures
    if not ok:
        _failures += 1
    line = f"  [{'PASS' if ok else 'FAIL'}] {name}"
    if detail:
        line += f"  - {detail}"
    print(line)


class FakeAnalyzeLLM:
    """Returns a canned analyze-shaped JSON, mimicking OllamaClient.generate_json."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def warmup(self) -> None:  # pragma: no cover
        pass

    def generate_json(self, prompt: str) -> dict:
        return dict(self._payload)


class StubRetriever:
    """Returns a fixed, known set of chunks — so citation-verification and the
    current-law note are tested deterministically, independent of the fake embedder's
    (deliberately weak) ranking."""

    def __init__(self, chunks) -> None:
        from app.rag.models import RetrievedChunk

        self._results = [RetrievedChunk(chunk=c, score=1.0) for c in chunks]

    def retrieve(self, query, **kwargs):
        return list(self._results)


def _ipc420_and_bns318():
    from app.rag.models import Chunk

    return [
        Chunk.create(
            text="IPC Section 420: Cheating and dishonestly inducing delivery of property. Punishment: 7 years and fine.",
            source_type="statute", ref="IPC-420", act="IPC", section="420",
            code_status="repealed", verification="official", source_authority="India Code (indiacode.nic.in)",
        ),
        Chunk.create(
            text="BNS Section 318(4): Cheating and dishonestly inducing delivery of property. Punishment: up to 7 years and fine.",
            source_type="statute", ref="BNS-318", act="BNS", section="318(4)",
            code_status="current", verification="curated", source_authority="Curated starter",
        ),
    ]


SAMPLE_DOC = (
    "LEGAL NOTICE\n\n"
    "To: Mr. Sharma\n"
    "You are hereby informed that you cheated the complainant and dishonestly induced "
    "delivery of Rs. 2,00,000 by false promise, an offence of cheating under Section 420. "
    "You are called upon to repay the said amount within 15 days of receipt of this notice, "
    "failing which appropriate criminal and civil proceedings will be initiated.\n"
)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="nyaysetu_doc_"))
    print(f"Document smoke test - temp index at {tmp}\n" + "=" * 72)
    try:
        from app.rag.retriever import HybridRetriever
        from app.rag.vector_store import VectorStore
        from app.services.document_service import DocumentService
        from app.utils.extract import ExtractionError, extract_text

        setup_index(tmp)
        analyze_payload = {
            "document_type": "Legal notice for cheating (Section 420)",
            "summary": "A demand notice alleging cheating and demanding repayment.",
            "key_points": ["You are accused of cheating under Section 420", "Repayment demanded"],
            "deadlines": ["Repay within 15 days of receipt"],
            "action": "Consult a lawyer before the 15-day period expires.",
            "law_references": ["IPC Section 420"],
        }

        # --- extract_text util ---
        print("\nText extraction:")
        check("reads .txt bytes", extract_text("notice.txt", b"Hello legal world") == "Hello legal world")
        check("reads .md bytes", extract_text("a.md", b"# Title\nbody") .startswith("# Title"))
        try:
            extract_text("evil.exe", b"\x00\x01")
            check("rejects unsupported ext", False)
        except ExtractionError:
            check("rejects unsupported ext", True)
        try:
            extract_text("empty.txt", b"   ")
            check("rejects empty file", False)
        except ExtractionError:
            check("rejects empty file", True)

        # --- Scenario A: end-to-end over the REAL hybrid retriever (integration) ---
        # The fake embedder is deliberately weak, so we assert only what doesn't depend
        # on its ranking; citation-verification logic is proven deterministically in A2.
        print("\nScenario A - end-to-end analysis over the real retriever:")
        svc_a = DocumentService(llm=FakeAnalyzeLLM(analyze_payload), retriever=HybridRetriever())
        a = svc_a.analyze(SAMPLE_DOC)
        check("not abstained", a.abstained is False)
        check("document_type set", bool(a.document_type), a.document_type)
        check("has key_points", len(a.key_points) > 0)
        check("captured deadline", len(a.deadlines) > 0, "; ".join(a.deadlines))
        check("has citations", len(a.citations) > 0, f"{len(a.citations)} citations")
        # current-law note is retrieval-dependent (fake embedder); A2 verifies it deterministically.
        print(f"       (current_law_note from real retrieval: {(a.current_law_note or 'none')[:50]})")

        # --- Scenario A2: grounded + verified (deterministic stub retrieving IPC 420 + BNS 318) ---
        print("\nScenario A2 - grounded, verified, current-law note (deterministic):")
        svc_a2 = DocumentService(llm=FakeAnalyzeLLM(analyze_payload), retriever=StubRetriever(_ipc420_and_bns318()))
        a2 = svc_a2.analyze(SAMPLE_DOC)
        check("citation verified (IPC 420 present)", a2.citation_verified is True)
        check("confidence high", a2.confidence == "high", a2.confidence)
        check("current_law_note IPC->BNS", bool(a2.current_law_note) and "BNS Section 318" in (a2.current_law_note or ""),
              (a2.current_law_note or "")[:60])

        # --- Scenario B: hallucinated law reference is caught ---
        print("\nScenario B - hallucinated section caught:")
        bad = {**analyze_payload, "law_references": ["BNS Section 9999"]}
        svc_b = DocumentService(llm=FakeAnalyzeLLM(bad), retriever=StubRetriever(_ipc420_and_bns318()))
        b = svc_b.analyze(SAMPLE_DOC)
        check("citation_verified is False", b.citation_verified is False)
        check("confidence downgraded to low", b.confidence == "low", b.confidence)
        check("escalation surfaced", bool(b.escalation))

        print("=" * 72)
        if _failures == 0:
            print("ALL CHECKS PASSED")
        else:
            print(f"{_failures} CHECK(S) FAILED")
        return 1 if _failures else 0
    finally:
        VectorStore.reset()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
