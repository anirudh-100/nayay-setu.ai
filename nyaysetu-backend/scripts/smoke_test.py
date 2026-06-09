"""Offline end-to-end smoke test for the RAG pipeline — no model downloads, no Ollama.

It exercises the REAL stack — loaders, Qdrant (embedded), BM25, RRF fusion, current-law
cross-reference expansion, citation verification, and response assembly — substituting
only the two heavy externals:
  - the embedding model      -> a deterministic hashing vectorizer (FakeEmbedder)
  - the Ollama LLM           -> canned JSON (FakeLLM)

So it proves the orchestration logic is wired correctly even before you install torch,
download InLegalBERT, or run Ollama. It builds a small temp index (statutes + guides),
runs three scenarios, and asserts the expected behaviour.

Usage:
    python scripts/smoke_test.py
"""
from __future__ import annotations

import hashlib
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402

_TOKEN_RE = re.compile(r"\w+")


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeEmbedder:
    """Deterministic bag-of-words hashing vectorizer — stands in for InLegalBERT.

    Not semantically smart, but stable and dependency-free, so dense retrieval
    behaves consistently for the test (and the lexical/BM25 half is fully real)."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim
        self.model_name = "fake-hashing"
        self.max_seq_length = 256

    @property
    def dimension(self) -> int:
        return self._dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self._dim, dtype=np.float32)
        for tok in _TOKEN_RE.findall(text.lower()):
            bucket = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self._dim
            v[bucket] += 1.0
        n = float(np.linalg.norm(v))
        if n > 0:
            v /= n
        return v

    def encode_passages(self, texts, batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        return np.vstack([self._vec(t) for t in texts]).astype(np.float32)

    def encode_queries(self, texts, batch_size: int = 32) -> np.ndarray:
        return self.encode_passages(texts)

    def encode_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class FakeLLM:
    """Returns a canned structured response, mimicking OllamaClient.generate_json."""

    def __init__(self, ref: str, reasoning: str, answer: str, action: str) -> None:
        self._payload = {
            "answer": answer,
            "law_reference": ref,
            "action": action,
            "confidence": "high",
            "reasoning": reasoning,
        }

    def warmup(self) -> None:  # pragma: no cover
        pass

    def generate_json(self, prompt: str) -> dict:
        return dict(self._payload)


class EmptyRetriever:
    """Forces the abstention branch (no retrieval results)."""

    def retrieve(self, query, **kwargs):
        return []


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
PASS, FAIL = "PASS", "FAIL"
_failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global _failures
    if not ok:
        _failures += 1
    tag = PASS if ok else FAIL
    line = f"  [{tag}] {name}"
    if detail:
        line += f"  - {detail}"
    print(line)


def setup_index(tmp: Path):
    """Build a small temp index over statutes + guides using the fake embedder."""
    from app.rag import pipeline
    from app.rag.embedder import Embedder
    from app.rag.lexical_store import LexicalStore
    from app.rag.vector_store import VectorStore

    # Point all index artifacts at the temp dir; disable the (download-heavy) reranker.
    settings.index_dir = tmp
    settings.qdrant_url = ""
    settings.use_reranker = False

    # Inject the fake embedder as the singleton.
    Embedder._instance = FakeEmbedder()  # type: ignore[assignment]
    VectorStore.reset()
    LexicalStore.reset()

    # Subset load_all() to keep the build fast + deterministic (skip the 10k QA pairs).
    orig_load_all = pipeline.load_all

    def subset():
        return [c for c in orig_load_all() if c.source_type in ("statute", "guide")]

    pipeline.load_all = subset  # type: ignore[assignment]
    stats = pipeline.build_index()
    print(f"  built temp index: {stats}\n")
    return stats


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="nyaysetu_smoke_"))
    print(f"Smoke test - temp index at {tmp}\n" + "=" * 72)

    try:
        from app.rag.retriever import HybridRetriever
        from app.rag.vector_store import VectorStore
        from app.services.rag_service import RAGService

        setup_index(tmp)
        query = "What is the punishment for cheating under IPC 420?"

        # --- raw retrieval sanity ---
        print("Retrieval (real hybrid: fake-dense + real BM25 + RRF):")
        results = HybridRetriever().retrieve(query, top_k=6)
        secs = [f"{r.chunk.act}:{r.chunk.section}" for r in results if r.chunk.section]
        check("retrieves something", len(results) > 0, f"{len(results)} chunks")
        check("finds IPC 420", any(r.chunk.act == "IPC" and r.chunk.section == "420" for r in results),
              " ".join(secs[:8]))

        # --- Scenario A: grounded, current-law-aware, verified ---
        print("\nScenario A - grounded answer citing current BNS law:")
        svc_a = RAGService(
            llm=FakeLLM("BNS Section 318", "Used BNS Section 318 from context",
                        "Cheating is punishable...", "File a police complaint."),
            retriever=HybridRetriever(),
        )
        a = svc_a.answer(query)
        check("not abstained", a.abstained is False)
        check("has citations", len(a.citations) > 0, f"{len(a.citations)} citations")
        check("cross-ref pulled BNS 318", any("BNS Section 318" in c.label for c in a.citations))
        check("current_law_note present (IPC->BNS)", bool(a.current_law_note),
              (a.current_law_note or "")[:60])
        check("citation verified", a.citation_verified is True)
        check("confidence high", a.confidence == "high", a.confidence)

        # --- Scenario B: hallucinated citation is caught ---
        print("\nScenario B - hallucinated section is caught by verification:")
        svc_b = RAGService(
            llm=FakeLLM("IPC Section 999", "Used IPC Section 999 from context",
                        "...", "..."),
            retriever=HybridRetriever(),
        )
        b = svc_b.answer(query)
        check("citation_verified is False", b.citation_verified is False)
        check("confidence downgraded to low", b.confidence == "low", b.confidence)
        check("escalation surfaced", bool(b.escalation))

        # --- Scenario C: abstention on empty retrieval ---
        print("\nScenario C - abstention when nothing is retrieved:")
        svc_c = RAGService(llm=FakeLLM("x", "x", "x", "x"), retriever=EmptyRetriever())
        c = svc_c.answer("asldkfj qwpoieu nonsense zzz")
        check("abstained is True", c.abstained is True)
        check("escalation surfaced", bool(c.escalation))
        check("confidence low", c.confidence == "low", c.confidence)

        print("=" * 72)
        if _failures == 0:
            print("ALL CHECKS PASSED")
        else:
            print(f"{_failures} CHECK(S) FAILED")
        return 1 if _failures else 0

    finally:
        VectorStore.reset()  # release the embedded Qdrant file lock before cleanup
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
