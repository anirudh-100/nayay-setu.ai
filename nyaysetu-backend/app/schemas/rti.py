"""Request/response models for the /draft/rti endpoint (RTI application drafting).

The citizen-empowerment counterpart to a lawyer's drafting desk: a person describes,
in plain words, what they want to know from the government; NyaySetu returns a ready-to-
send Right to Information application plus the procedural scaffolding to file it — fee,
where to send it, the statutory time limits, and the appeal ladder if they're ignored.

Trust contract: the legal scaffolding (RTI Act sections, fee, timelines, appeals) is
drawn from ``app.knowledge.rti`` — hand-authored from the official Act, never invented.
The language model's only job is to turn the request into specific, answerable questions.
The drafted letter is the *applicant's own document*, clearly labelled a draft to review.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.rag.models import Citation
from app.schemas.ask import Confidence

GovLevel = Literal["central", "state"]

RTI_DISCLAIMER = (
    "This is a draft Right to Information application to help you file one yourself — "
    "review it before sending, and confirm your State's RTI fee and filing address. "
    "It is informational, not legal advice."
)


class RTIDraftRequest(BaseModel):
    # The heart of it: what the person wants to know, in their own words.
    subject: str = Field(..., min_length=5, max_length=2000)
    # Which government department/body holds the information (optional — we draft with a
    # clear placeholder if unknown, and tell the user to fill it in).
    public_authority: Optional[str] = Field(default=None, max_length=300)
    # Drives fee guidance + filing portal. Defaults to central.
    level: GovLevel = "central"
    # Personal details for the application letter (optional — placeholders used if absent).
    applicant_name: Optional[str] = Field(default=None, max_length=120)
    applicant_address: Optional[str] = Field(default=None, max_length=500)
    # BPL applicants are exempt from the fee (RTI Act s.7(5)).
    is_bpl: bool = False
    language: str = Field(default="en", max_length=8)


class FilingInfo(BaseModel):
    where_to_file: str
    fee: str
    is_bpl_exempt: bool
    portal: Optional[str] = None


class RTIDraftResponse(BaseModel):
    # The ready-to-send application letter (the main deliverable).
    application_text: str
    # A short subject line for the application.
    subject_line: str
    # The specific, answerable questions the application asks (shown/editable in the UI).
    questions: list[str] = Field(default_factory=list)
    # The authority the letter is addressed to (echoed, or a placeholder prompt).
    public_authority: str

    # --- Procedural scaffolding (all from app.knowledge.rti — deterministic) ---
    filing: FilingInfo
    timeline: list[str] = Field(default_factory=list)
    appeals: list[str] = Field(default_factory=list)
    tips: list[str] = Field(default_factory=list)

    # --- Trust contract (shared shape with /ask and /analyze) ---
    confidence: Confidence
    citations: list[Citation] = Field(default_factory=list)
    citation_verified: bool = True
    disclaimer: str = RTI_DISCLAIMER
    response_time_ms: int
