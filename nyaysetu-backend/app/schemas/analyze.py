"""Request/response models for the /analyze endpoint (document understanding).

A citizen uploads or pastes a document they received — an FIR, a legal notice, a
summons, a rent agreement — and asks "what is this, and what do I do?". The response
mirrors the trust contract of /ask (citations, abstention, current-law note,
disclaimer) but is shaped around a *document*: what it is, its key points, any
deadlines, and the relevant law — every legal claim grounded in retrieved sources.
"""
from typing import Optional

from pydantic import BaseModel, Field

from app.rag.models import Citation
from app.schemas.ask import DISCLAIMER, Confidence

# Reasonable bounds: long enough for a multi-page notice, short enough to bound cost.
MIN_DOC_CHARS = 20
MAX_DOC_CHARS = 40_000


class AnalyzeRequest(BaseModel):
    document_text: str = Field(..., min_length=MIN_DOC_CHARS, max_length=MAX_DOC_CHARS)
    # Optional focused question, e.g. "how long do I have to respond?".
    question: Optional[str] = Field(default=None, max_length=2000)
    language: str = Field(default="en", max_length=8)


class AnalyzeResponse(BaseModel):
    # What kind of document this is, in plain words (e.g. "Police summons under CrPC").
    document_type: str
    # A short plain-language summary of what the document says.
    summary: str
    # The most important points a non-lawyer should understand.
    key_points: list[str] = Field(default_factory=list)
    # Any dates/deadlines detected (e.g. "Appear on 12 July 2026"). Empty if none found.
    deadlines: list[str] = Field(default_factory=list)
    # The single most important next step.
    action: str

    # --- Trust contract (shared with /ask) ---
    confidence: Confidence
    citations: list[Citation] = Field(default_factory=list)
    current_law_note: Optional[str] = None
    citation_verified: bool = True
    abstained: bool = False
    escalation: Optional[str] = None
    disclaimer: str = DISCLAIMER
    response_time_ms: int
