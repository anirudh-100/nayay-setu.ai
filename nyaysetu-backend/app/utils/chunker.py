"""Word-based chunking with overlap.

Token counts are approximated as word counts. The all-MiniLM-L6-v2 model has
a 256-wordpiece limit, so we keep chunks well under that in practice.
"""
from __future__ import annotations

import re
from typing import Iterable


_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    words = _normalize(text).split(" ")
    if not words or words == [""]:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_documents(docs: Iterable[tuple[str, str]], chunk_size: int, overlap: int) -> list[dict]:
    """Chunk (source, text) pairs into a flat list of chunk records."""
    out: list[dict] = []
    for source, text in docs:
        for i, body in enumerate(chunk_text(text, chunk_size, overlap)):
            out.append({"source": source, "chunk_id": i, "text": body})
    return out
