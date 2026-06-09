"""Cross-encoder reranker — the precision pass over fused candidates.

Hybrid retrieval (dense + BM25 → RRF) is good at *recall*: it pulls a broad set of
plausibly-relevant chunks. But its ordering is approximate. A cross-encoder reads
the query and each candidate **together** (not as separate vectors) and scores true
relevance — far more accurate than either retriever's ranking. The standard, proven
pattern is: retrieve ~30–50 cheap candidates, then rerank and keep the top ~5–8.

This is where most of the "answer quality" gain after legal embeddings comes from:
the LLM only ever sees a handful of chunks, so getting the *right* handful to the
top directly determines whether the answer is grounded in the correct section.

Default model ``cross-encoder/ms-marco-MiniLM-L-6-v2`` is small and CPU-friendly.
For multilingual content later, swap to ``BAAI/bge-reranker-v2-m3`` via config
(``RERANK_MODEL``) — no code change. Fully local; downloads once, then offline.
"""
from __future__ import annotations

from threading import Lock
from typing import Sequence

from app.config import settings
from app.rag.models import RetrievedChunk
from app.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Thread-safe singleton cross-encoder."""

    _instance: "Reranker | None" = None
    _lock = Lock()

    def __init__(self, model_name: str | None = None, *, device: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self._model_name = model_name or getattr(settings, "rerank_model", None) or _DEFAULT_MODEL
        logger.info("Loading reranker %r (device=%s)", self._model_name, device or "auto")
        self._model = CrossEncoder(self._model_name, device=device)

    @classmethod
    def instance(cls) -> "Reranker":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        top_k: int,
        batch_size: int = 32,
    ) -> list[RetrievedChunk]:
        """Score each candidate against the query and return the best ``top_k``.

        Sets ``rerank_score`` and promotes it to the final ``score`` on each
        returned chunk, so downstream code can sort/threshold on one field.
        """
        if not candidates:
            return []

        pairs = [(query, rc.chunk.text) for rc in candidates]
        scores = self._model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

        for rc, score in zip(candidates, scores):
            rc.rerank_score = float(score)
            rc.score = float(score)

        ranked = sorted(candidates, key=lambda rc: rc.rerank_score or 0.0, reverse=True)
        return ranked[:top_k]
