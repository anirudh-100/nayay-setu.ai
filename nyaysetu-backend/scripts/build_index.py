"""Build (or rebuild) the persisted RAG indexes.

Run this once after changing any data under ``data/`` (IPC CSV, IndicLegalQA JSON,
or guides in ``data/corpus/``). It loads + tags + embeds every source and writes:
  - a Qdrant collection (dense vectors + metadata payloads), and
  - a BM25 pickle (lexical index).

Usage:
    python scripts/build_index.py

The API must not be running against the same embedded Qdrant path while this runs
(embedded Qdrant holds a file lock). Build first, then start the server.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow running as a plain script (python scripts/build_index.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.rag.pipeline import build_index  # noqa: E402
from app.utils.logger import get_logger  # noqa: E402

logger = get_logger("build_index")


def main() -> int:
    logger.info("Building indexes into %s", settings.index_dir)
    logger.info("Embedding model: %s | reranker: %s", settings.embedding_model, settings.rerank_model)
    started = time.perf_counter()
    try:
        stats = build_index()
    except RuntimeError as e:
        logger.error("Build failed: %s", e)
        return 1
    elapsed = time.perf_counter() - started
    logger.info("Done in %.1fs — %s", elapsed, stats)
    print(f"\n[OK] Index built in {elapsed:.1f}s: {stats}")
    print(f"     Location: {settings.index_dir}")
    print("     Start the API with:  uvicorn app.main:app --reload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
