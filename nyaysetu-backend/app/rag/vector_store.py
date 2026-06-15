"""Persisted dense vector store — Qdrant in embedded (local) mode.

Why Qdrant over the previous raw FAISS:
  - **Persistence built in.** The old engine rebuilt the whole index in RAM on
    every startup (slow, and impossible to scale past a toy corpus). Qdrant writes
    to disk and reloads instantly.
  - **Metadata filtering at query time.** Legal retrieval needs to filter by act,
    jurisdiction, language, source type, and (later) code status / date. Qdrant
    does this inside the ANN search, not as a slow post-filter.
  - **Payload storage.** Each vector carries its full ``Chunk`` payload, so search
    returns reconstructed chunks with citations — no separate metadata file to keep
    in sync (the old ``metadata.json`` drift risk is gone).

Fully local & free: ``QdrantClient(path=...)`` runs embedded — no server, no
Docker, no network. (Set ``QDRANT_URL`` later to point at a real server in prod;
the rest of the code is unchanged.) Embedded mode holds a file lock, so the
build script and the running API must not open it at the same time — fine for our
build-then-serve workflow.
"""
from __future__ import annotations

import uuid
from threading import Lock
from typing import Iterable, Optional, Sequence

import numpy as np

from app.config import settings
from app.rag.models import Chunk, RetrievedChunk
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Stable namespace so a given chunk.id always maps to the same Qdrant point UUID.
# (Qdrant point ids must be ints or UUIDs; our ids are content hashes like
# "statute-ab12…", so we derive a deterministic UUID and keep the readable id in
# the payload.)
_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

_DEFAULT_COLLECTION = "nyaysetu_chunks"


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


class VectorStore:
    """Thread-safe singleton over an embedded (or remote) Qdrant collection."""

    _instance: "VectorStore | None" = None
    _lock = Lock()

    def __init__(self, collection: str | None = None) -> None:
        from qdrant_client import QdrantClient

        self._collection = collection or getattr(settings, "qdrant_collection", _DEFAULT_COLLECTION)

        url = getattr(settings, "qdrant_url", None)
        if url:
            self._client = QdrantClient(url=url)
            logger.info("Connected to Qdrant server at %s (collection=%s)", url, self._collection)
        else:
            path = str(getattr(settings, "qdrant_path", settings.index_dir / "qdrant"))
            self._client = QdrantClient(path=path)
            logger.info("Opened embedded Qdrant at %s (collection=%s)", path, self._collection)

    # ------------------------------------------------------------------ #
    # Singleton access
    # ------------------------------------------------------------------ #
    @classmethod
    def instance(cls) -> "VectorStore":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            if cls._instance is not None:
                cls._instance.close()
            cls._instance = None

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # pragma: no cover - best effort
            pass

    # ------------------------------------------------------------------ #
    # Index lifecycle (used by the ingestion pipeline)
    # ------------------------------------------------------------------ #
    def recreate(self, dim: int) -> None:
        """Drop and create a fresh collection sized for ``dim``-d cosine vectors."""
        from qdrant_client.models import Distance, VectorParams

        self._client.recreate_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logger.info("Recreated Qdrant collection %r (dim=%d, cosine)", self._collection, dim)

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray, *, batch_size: int = 256) -> None:
        """Index ``chunks`` with their ``vectors`` (parallel arrays, same length)."""
        from qdrant_client.models import PointStruct

        if len(chunks) != len(vectors):
            raise ValueError(f"chunks ({len(chunks)}) and vectors ({len(vectors)}) length mismatch")
        if len(chunks) == 0:
            return

        points = [
            PointStruct(id=_point_id(c.id), vector=v.tolist(), payload=c.model_dump())
            for c, v in zip(chunks, vectors)
        ]
        for start in range(0, len(points), batch_size):
            self._client.upsert(
                collection_name=self._collection,
                points=points[start : start + batch_size],
            )
        logger.info("Upserted %d points into %r", len(points), self._collection)

    def count(self) -> int:
        return self._client.count(collection_name=self._collection, exact=True).count

    def delete_by_filter(self, filters: dict) -> int:
        """Delete all points matching a payload filter (e.g. {"source_type": "judgment"}).

        Used for clean incremental re-ingestion: remove a source's existing points before
        re-upserting, so changed text (new content-hash ids) doesn't leave orphan chunks.
        Returns the number of points removed.
        """
        flt = self._build_filter(filters)
        if flt is None:
            return 0
        from qdrant_client.models import FilterSelector

        before = self.count()
        self._client.delete(collection_name=self._collection, points_selector=FilterSelector(filter=flt))
        removed = before - self.count()
        logger.info("Deleted %d points matching %s from %r", removed, filters, self._collection)
        return removed

    def fetch_by_reference(self, *, act: str, sections: list[str], limit: int = 4) -> list[Chunk]:
        """Look up exact chunks by act + section(s) — used for cross-reference expansion
        (e.g. pull the current BNS chunk for a retrieved repealed IPC section). This is a
        metadata scroll, not a similarity search, so it returns the precise sections asked for."""
        if not sections:
            return []
        records, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=self._build_filter({"act": act, "section": sections}),
            limit=limit,
            with_payload=True,
        )
        chunks: list[Chunk] = []
        for rec in records:
            try:
                chunks.append(Chunk(**rec.payload))
            except Exception:
                continue
        return chunks

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int,
        filters: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        """Cosine ANN search. ``filters`` is a plain dict of payload field -> value
        (or list of allowed values), e.g. ``{"language": "en", "act": ["BNS", "IPC"]}``.
        """
        hits = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=self._build_filter(filters),
            with_payload=True,
        )
        results: list[RetrievedChunk] = []
        for hit in hits:
            try:
                chunk = Chunk(**hit.payload)
            except Exception as e:  # payload schema drift — skip rather than crash a query
                logger.warning("Skipping malformed payload (point %s): %s", hit.id, e)
                continue
            results.append(
                RetrievedChunk(chunk=chunk, score=float(hit.score), dense_score=float(hit.score))
            )
        return results

    @staticmethod
    def _build_filter(filters: Optional[dict]):
        if not filters:
            return None
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

        conditions: list = []
        for field, value in filters.items():
            if isinstance(value, (list, tuple, set)):
                conditions.append(FieldCondition(key=field, match=MatchAny(any=list(value))))
            else:
                conditions.append(FieldCondition(key=field, match=MatchValue(value=value)))
        return Filter(must=conditions)
