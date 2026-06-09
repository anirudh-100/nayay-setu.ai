"""Hybrid retriever — the heart of the engine.

Pipeline (the proven legal-RAG recipe):

    query
      ├─▶ dense search   (InLegalBERT + Qdrant, metadata-filtered)  ─┐
      └─▶ lexical search (BM25, exact tokens like section numbers)  ─┤
                                                                     ▼
                          Reciprocal Rank Fusion (RRF)  ── combine both rankings
                                                                     ▼
                          cross-encoder rerank (50 → top-k)
                                                                     ▼
                          top-k RetrievedChunk (with all component scores)

Why RRF and not score-weighting: dense (cosine ∈ [-1,1]) and BM25 (unbounded) scores
live on totally different scales, so adding them directly is meaningless. RRF ignores
the raw scores and fuses *rank positions*, which is scale-free, robust, and the
standard fusion method for hybrid search. The reranker then re-scores the fused
shortlist on an absolute relevance scale we *can* threshold on (for abstention).

Each stage is optional/tunable via config so we can measure its contribution and
trade latency for quality.
"""
from __future__ import annotations

from typing import Optional

from app.config import settings
from app.rag.models import RetrievedChunk
from app.utils.logger import get_logger

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]],
    *,
    k: int = 60,
) -> list[RetrievedChunk]:
    """Fuse multiple ranked lists by RRF. ``k`` damps the contribution of low ranks
    (60 is the value from the original RRF paper and the common default).

    Chunks are merged by ``chunk.id``; the merged entry keeps every component score
    (dense_score / lexical_score) seen across lists, and ``score`` becomes the RRF
    total — useful for inspection, though the reranker usually overrides it next.
    """
    fused: dict[str, RetrievedChunk] = {}
    rrf_score: dict[str, float] = {}

    for results in result_lists:
        for rank, rc in enumerate(results):
            cid = rc.chunk.id
            rrf_score[cid] = rrf_score.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid not in fused:
                # First time we see this chunk — take it as the base record.
                fused[cid] = rc.model_copy(deep=True)
            else:
                # Seen via another retriever — fold in whichever component score it carries.
                existing = fused[cid]
                if rc.dense_score is not None:
                    existing.dense_score = rc.dense_score
                if rc.lexical_score is not None:
                    existing.lexical_score = rc.lexical_score

    for cid, rc in fused.items():
        rc.score = rrf_score[cid]

    return sorted(fused.values(), key=lambda rc: rc.score, reverse=True)


def _matches_filters(rc: RetrievedChunk, filters: Optional[dict]) -> bool:
    """Apply the same metadata filters Qdrant uses to BM25 hits (which are unfiltered),
    so both retrievers agree on the candidate universe."""
    if not filters:
        return True
    payload = rc.chunk.model_dump()
    for field, value in filters.items():
        actual = payload.get(field)
        if isinstance(value, (list, tuple, set)):
            if actual not in value:
                return False
        elif actual != value:
            return False
    return True


class HybridRetriever:
    """Stateless orchestrator over the dense, lexical, and rerank singletons."""

    def __init__(
        self,
        *,
        fetch_k: int | None = None,
        rerank_candidates: int | None = None,
        top_k: int | None = None,
        rrf_k: int | None = None,
        use_reranker: bool | None = None,
    ) -> None:
        # Candidates pulled from EACH retriever before fusion.
        self.fetch_k = fetch_k or getattr(settings, "fetch_k", 30)
        # Size of the fused shortlist handed to the (slow) reranker.
        self.rerank_candidates = rerank_candidates or getattr(settings, "rerank_candidates", 50)
        # Final number of chunks returned to the LLM.
        self.top_k = top_k or getattr(settings, "top_k", 6)
        self.rrf_k = rrf_k or getattr(settings, "rrf_k", 60)
        self.use_reranker = (
            use_reranker if use_reranker is not None else getattr(settings, "use_reranker", True)
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        from app.rag.embedder import Embedder
        from app.rag.lexical_store import LexicalStore
        from app.rag.vector_store import VectorStore

        final_k = top_k or self.top_k

        # 1. Dense (semantic) — filtered inside Qdrant.
        qvec = Embedder.instance().encode_query(query)
        dense = VectorStore.instance().search(qvec, top_k=self.fetch_k, filters=filters)

        # 2. Lexical (exact tokens) — filter in Python to match the dense universe.
        lexical = LexicalStore.instance().search(query, top_k=self.fetch_k)
        if filters:
            lexical = [rc for rc in lexical if _matches_filters(rc, filters)]

        logger.info("Retrieved dense=%d lexical=%d for query=%r", len(dense), len(lexical), query[:100])

        # 3. Fuse.
        fused = reciprocal_rank_fusion([dense, lexical], k=self.rrf_k)
        if not fused:
            return []

        # 4. Rerank the shortlist (or fall back to fused order).
        if self.use_reranker:
            from app.rag.reranker import Reranker

            shortlist = fused[: self.rerank_candidates]
            return Reranker.instance().rerank(query, shortlist, top_k=final_k)
        return fused[:final_k]
