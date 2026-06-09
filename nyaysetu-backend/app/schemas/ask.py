"""Request/response models for the /ask endpoint.

Extends the original contract (answer / law_reference / action / confidence /
reasoning / disclaimer / response_time_ms) with the trust-critical additions from
the research: **verifiable citations**, an **abstention flag**, and a **human
escalation** pointer (free legal aid) for low-confidence answers. Existing fields
are unchanged so the current frontend keeps working.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.rag.models import Citation

Confidence = Literal["high", "medium", "low"]

DISCLAIMER = (
    "This is for informational purposes only and not legal advice. "
    "Consult a qualified lawyer for guidance on your specific situation."
)

# Surfaced when the engine is unsure: point users to free, official legal aid
# rather than letting a low-confidence answer stand alone. (NALSA/DLSA provide
# free legal services across India; 15100 is the national legal-aid helpline.)
LEGAL_AID_ESCALATION = (
    "For free, official legal help, contact your District Legal Services Authority "
    "(DLSA) or call the NALSA legal-aid helpline at 15100."
)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    # Reserved for the multilingual phase; ignored by the engine today but accepted
    # so the frontend contract is stable.
    language: str = Field(default="en", max_length=8)


class AskResponse(BaseModel):
    answer: str
    law_reference: str
    action: str
    confidence: Confidence
    reasoning: str

    # --- Trust additions ---
    citations: list[Citation] = Field(default_factory=list)
    abstained: bool = False
    escalation: Optional[str] = None
    # Bridges old<->new law, e.g. "IPC Section 420 now corresponds to BNS Section 318(4)".
    current_law_note: Optional[str] = None
    # False when the answer cited a section that wasn't found in the retrieved sources
    # (a hallucination signal) — confidence is downgraded when this happens.
    citation_verified: bool = True

    disclaimer: str = DISCLAIMER
    response_time_ms: int
