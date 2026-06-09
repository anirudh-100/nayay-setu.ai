"""Sparse lexical index (BM25) — the keyword half of hybrid retrieval.

Dense vectors are great at *meaning* but weak at *exact tokens*. Legal queries are
full of exact tokens that must match precisely: section numbers ("420", "318"),
article numbers, case citations, party names, dates. BM25 nails those; dense
search often drifts to a conceptually-similar-but-wrong section. Running both and
fusing them (see ``retriever.py``) measurably beats either alone on legal text —
this is the single most important retrieval upgrade after using legal embeddings.

Implementation: ``rank_bm25`` (pure Python, no native deps, fully local). The index
is small and rebuilds in well under a second for tens of thousands of chunks, so we
persist the tokenized corpus + chunk payloads and reconstruct BM25 on load.

The tokenizer keeps digits and Unicode letters (``\w+``), so section numbers survive
and Indic-script text will tokenize sensibly when we add multilingual content.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from threading import Lock
from typing import Sequence

from app.config import settings
from app.rag.models import Chunk, RetrievedChunk
from app.utils.logger import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_DEFAULT_FILENAME = "bm25.pkl"


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class LexicalStore:
    """Thread-safe singleton BM25 index over the chunk corpus."""

    _instance: "LexicalStore | None" = None
    _lock = Lock()

    def __init__(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        self._chunks = chunks
        tokenized = [_tokenize(c.text) for c in chunks]
        # Guard against an all-empty corpus (BM25Okapi divides by avgdl).
        self._bm25 = BM25Okapi(tokenized) if any(tokenized) else None
        logger.info("Built BM25 index over %d chunks", len(chunks))

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    @staticmethod
    def _default_path() -> Path:
        return Path(getattr(settings, "index_dir")) / _DEFAULT_FILENAME

    @classmethod
    def build_and_save(cls, chunks: Sequence[Chunk], path: Path | None = None) -> None:
        path = path or cls._default_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [c.model_dump() for c in chunks]
        with path.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved BM25 corpus (%d chunks) to %s", len(payload), path)

    @classmethod
    def from_disk(cls, path: Path | None = None) -> "LexicalStore":
        path = path or cls._default_path()
        if not path.exists():
            raise FileNotFoundError(
                f"BM25 corpus not found at {path}. Run: python scripts/build_index.py"
            )
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        chunks = [Chunk(**d) for d in payload]
        return cls(chunks)

    # ------------------------------------------------------------------ #
    # Singleton access
    # ------------------------------------------------------------------ #
    @classmethod
    def instance(cls) -> "LexicalStore":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls.from_disk()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def search(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        if self._bm25 is None or not self._chunks:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        # Top-k by score, descending; skip zero/negative (no token overlap).
        ranked = sorted(enumerate(scores), key=lambda kv: kv[1], reverse=True)[:top_k]
        results: list[RetrievedChunk] = []
        for idx, score in ranked:
            if score <= 0:
                continue
            results.append(
                RetrievedChunk(
                    chunk=self._chunks[idx],
                    score=float(score),
                    lexical_score=float(score),
                )
            )
        return results

    def __len__(self) -> int:
        return len(self._chunks)
