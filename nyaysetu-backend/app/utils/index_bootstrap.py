"""Fetch the prebuilt RAG index at startup when it's missing (production).

The 105 MB Qdrant + BM25 index is too slow to rebuild on a server (~75 min embed
pass) and too big to keep in git, so in production we ship it as a single zip
artifact (e.g. a GitHub Release) and download it on first boot. Behaviour:

  - index already present  -> no-op (idempotent across restarts),
  - missing + INDEX_URL set -> download the zip and extract it into the index dir,
  - missing + no INDEX_URL  -> log a clear warning and carry on (the app still boots;
                               retrieval just abstains until an index exists).

The zip must contain the *contents* of models/index at its root (``qdrant/`` and
``bm25.pkl``), so extracting into the index dir reconstructs the expected layout.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _present() -> bool:
    return settings.qdrant_path.exists() and settings.bm25_file.exists()


def ensure_index() -> bool:
    """Make sure a usable index exists locally. Returns True if one is present
    (already, or after a successful download); False if it couldn't be provisioned."""
    if _present():
        return True

    url = (settings.index_url or "").strip()
    if not url:
        logger.warning(
            "Index missing at %s and INDEX_URL not set — retrieval will abstain until "
            "an index is built (scripts/build_index.py) or INDEX_URL is configured.",
            settings.index_dir,
        )
        return False

    index_dir = Path(settings.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Index missing; downloading prebuilt index from %s ...", url)
    try:
        import httpx

        buf = io.BytesIO()
        with httpx.stream("GET", url, follow_redirects=True, timeout=600) as r:
            r.raise_for_status()
            for chunk in r.iter_bytes():
                buf.write(chunk)
        buf.seek(0)
        with zipfile.ZipFile(buf) as zf:
            zf.extractall(index_dir)
    except Exception:
        logger.exception("Failed to download/extract the index from %s", url)
        return False

    if _present():
        logger.info("Index downloaded and ready at %s", index_dir)
        return True
    logger.error(
        "Downloaded archive did not yield %s + %s — check the zip contains qdrant/ and "
        "bm25.pkl at its root (zip the CONTENTS of models/index, not the folder).",
        settings.qdrant_path, settings.bm25_file,
    )
    return False
