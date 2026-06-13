"""Extract plain text from an uploaded document (txt / md / pdf).

Document understanding starts with getting clean text out of whatever the user
uploaded. We deliberately support a small, safe set of formats — plain text and PDF
— and fail with a clear message otherwise, rather than silently mis-reading a file.

Everything is local: PDF text is pulled with ``pypdf`` (no network, no OCR service).
Scanned/image-only PDFs won't yield text — we detect that and say so, instead of
returning an empty analysis that looks like a real answer.
"""
from __future__ import annotations

import io

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Generous cap so a huge upload can't blow up embedding/LLM cost. ~40k chars is well
# beyond a typical FIR/notice/agreement page count while staying safe.
MAX_CHARS = 40_000

_TEXT_EXTS = {".txt", ".md", ".markdown", ".text"}
_PDF_EXTS = {".pdf"}


class ExtractionError(ValueError):
    """Raised when a document can't be turned into usable text."""


def _ext(filename: str) -> str:
    name = (filename or "").lower()
    dot = name.rfind(".")
    return name[dot:] if dot != -1 else ""


def extract_text(filename: str, data: bytes) -> str:
    """Return clean UTF-8 text from an uploaded file, or raise ``ExtractionError``."""
    ext = _ext(filename)

    if ext in _TEXT_EXTS or ext == "":
        text = data.decode("utf-8", errors="replace")
    elif ext in _PDF_EXTS:
        text = _extract_pdf(data)
    else:
        raise ExtractionError(
            f"Unsupported file type '{ext or 'unknown'}'. Upload a PDF or a text (.txt/.md) file, "
            "or paste the text directly."
        )

    text = text.strip()
    if not text:
        raise ExtractionError(
            "Couldn't read any text from this file. If it's a scanned/photo PDF, the text isn't "
            "selectable — please type or paste the key parts instead."
        )
    if len(text) > MAX_CHARS:
        logger.info("Truncating extracted text from %d to %d chars", len(text), MAX_CHARS)
        text = text[:MAX_CHARS]
    return text


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as e:  # corrupt / encrypted / not really a PDF
        raise ExtractionError(f"Couldn't open this PDF ({type(e).__name__}). It may be corrupted or password-protected.") from e

    parts: list[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:  # one bad page shouldn't sink the whole document
            continue
    return "\n".join(parts)
