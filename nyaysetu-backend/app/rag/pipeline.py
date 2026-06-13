"""Ingestion pipeline: load → dedupe → embed → persist (both indexes).

Run once (via ``scripts/build_index.py``) whenever the data changes. It writes a
persisted Qdrant collection (dense) and a BM25 pickle (lexical); the API then loads
both at startup in milliseconds instead of re-embedding the whole corpus on every
boot like the old engine did.

Dedup is free here: chunk ids are content hashes, so identical text from two sources
collapses to one vector automatically.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass

from app.config import settings
from app.rag.embedder import Embedder
from app.rag.lexical_store import LexicalStore
from app.rag.loaders import load_all
from app.rag.models import Chunk
from app.rag.vector_store import VectorStore
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BuildStats:
    loaded: int
    indexed: int
    duplicates: int
    dimension: int

    def __str__(self) -> str:
        return (
            f"loaded={self.loaded} indexed={self.indexed} "
            f"duplicates={self.duplicates} dim={self.dimension}"
        )


def _dedupe(chunks: list[Chunk]) -> tuple[list[Chunk], int]:
    seen: dict[str, Chunk] = {}
    for c in chunks:
        seen.setdefault(c.id, c)
    deduped = list(seen.values())
    return deduped, len(chunks) - len(deduped)


def build_index() -> BuildStats:
    """Full rebuild of both indexes from all configured sources."""
    chunks = load_all()
    if not chunks:
        raise RuntimeError(
            "No chunks loaded. Check data dirs (data/ipc, data/indiclegalqa, data/corpus)."
        )

    chunks, duplicates = _dedupe(chunks)
    logger.info("Embedding %d chunks (%d duplicates dropped)...", len(chunks), duplicates)

    embedder = Embedder.instance()
    vectors = embedder.encode_passages([c.text for c in chunks])

    # Clean slate: embedded Qdrant's recreate_collection can leave stale points behind
    # in local mode (observed: a prior build's chunks survived a rebuild), which would
    # let outdated/curated chunks compete with current official ones. For embedded mode
    # we physically delete the on-disk collection before rebuilding. (Remote Qdrant
    # relies on recreate_collection.)
    if not getattr(settings, "qdrant_url", ""):
        VectorStore.reset()
        shutil.rmtree(settings.qdrant_path, ignore_errors=True)

    # Dense index (fresh collection each rebuild — deterministic, no stale points).
    store = VectorStore.instance()
    store.recreate(dim=embedder.dimension)
    store.upsert(chunks, vectors)

    # Lexical index.
    LexicalStore.build_and_save(chunks)

    stats = BuildStats(
        loaded=len(chunks) + duplicates,
        indexed=len(chunks),
        duplicates=duplicates,
        dimension=embedder.dimension,
    )
    logger.info("Index build complete: %s", stats)
    return stats
