"""Incrementally add judgment chunks to the existing index — no full re-embed.

A full rebuild re-embeds all ~11.9k chunks (~75 min on CPU) just to add ~289 judgment
chunks, and won't survive the machine sleeping. This instead embeds ONLY the judgment
chunks and upserts them into the existing Qdrant collection, then rebuilds the cheap BM25
lexical index over ALL chunks so judgments are both densely and lexically searchable.

Safe because chunk ids are content hashes (idempotent upsert) and the rest of the corpus
is unchanged since the last full build. Requires the API server stopped (embedded Qdrant
holds a file lock). Re-runnable.

Usage:
    python scripts/index_add_judgments.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    from app.rag.embedder import Embedder
    from app.rag.lexical_store import LexicalStore
    from app.rag.loaders import load_all
    from app.rag.pipeline import _dedupe
    from app.rag.vector_store import VectorStore
    from app.utils.logger import get_logger

    log = get_logger("index_add_judgments")

    all_chunks, dupes = _dedupe(load_all())
    judgments = [c for c in all_chunks if c.source_type == "judgment"]
    log.info("Corpus: %d chunks (%d dupes dropped); %d judgment chunks to add.",
             len(all_chunks), dupes, len(judgments))
    if not judgments:
        print("FAIL: no judgment chunks found (is data/judgments/ populated?)")
        return 1

    # 1. Embed ONLY the new judgment chunks (the expensive step, but ~40x smaller).
    embedder = Embedder.instance()
    log.info("Embedding %d judgment chunks...", len(judgments))
    vectors = embedder.encode_passages([c.text for c in judgments])

    # 2. Replace any existing judgment points (clean re-ingest: changed text => new
    #    content-hash ids, so delete-then-upsert avoids orphaned stale chunks), then
    #    upsert into the EXISTING dense collection (non-judgment vectors are untouched).
    store = VectorStore.instance()
    before = store.count()
    store.delete_by_filter({"source_type": "judgment"})
    store.upsert(judgments, vectors)
    after = store.count()
    log.info("Dense index: %d -> %d points (judgment chunks now: %d).", before, after, len(judgments))

    # 3. Rebuild the lexical (BM25) index over ALL chunks — cheap, keeps the two in sync.
    LexicalStore.build_and_save(all_chunks)

    VectorStore.reset()  # release the embedded-Qdrant file lock

    print("=" * 64)
    print(f"[OK] dense {before} -> {after} points ({len(judgments)} judgment chunks); "
          f"BM25 rebuilt over {len(all_chunks)} chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
