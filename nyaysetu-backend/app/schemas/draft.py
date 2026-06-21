"""Request/response models for the generic drafting engine (/draft).

NyaySetu's "drafting desk" for citizens: pick a document type (a *journey* — RTI,
consumer complaint, police complaint, …), describe the situation in plain words, and
get a ready-to-file document plus the procedural scaffolding to use it (where to file,
fee, time limits, what to do if ignored).

One engine serves every journey. A journey is defined once in ``app.knowledge.drafting``
— its input fields, the official legal facts it cites, and how it renders the document —
so adding a new kind of letter is a single knowledge module, not a new endpoint.

Trust contract (shared with /ask and /analyze): the legal scaffolding each journey cites
is hand-authored from the official Act and marked ``curated``/``official`` with its
source — never LLM-invented. The model only turns the citizen's situation into the
document's specific content; the document and all legal facts are assembled
deterministically, with a graceful fallback if the model is unavailable.
"""
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

from app.rag.models import Citation
from app.schemas.ask import Confidence

FieldKind = Literal["text", "textarea", "select", "checkbox"]
SectionTone = Literal["default", "info", "warn"]

DRAFT_DISCLAIMER = (
    "This is a draft to help you act for yourself — review it, fill in the blanks, and "
    "confirm the details (fee, exact office/forum, and any local rules) before using it. "
    "It is informational, not legal advice."
)
DRAFT_DISCLAIMER_HI = (
    "यह आपकी अपनी मदद के लिए एक मसौदा है — उपयोग से पहले इसे ध्यान से पढ़ें, रिक्त स्थान भरें, "
    "और विवरण (शुल्क, सही कार्यालय/मंच, और स्थानीय नियम) की पुष्टि कर लें। यह केवल जानकारी है, "
    "कानूनी सलाह नहीं।"
)


class FieldOption(BaseModel):
    value: str
    label: str


class FieldSpec(BaseModel):
    """Describes one input a journey needs — lets the frontend render forms generically."""

    key: str
    label: str
    kind: FieldKind = "text"
    required: bool = False
    placeholder: Optional[str] = None
    help: Optional[str] = None
    options: list[FieldOption] = Field(default_factory=list)  # for kind == "select"
    default: Optional[str] = None


class JourneyInfo(BaseModel):
    """Public description of a journey, returned by GET /draft/journeys."""

    id: str
    title: str            # menu label, e.g. "Demand information (RTI)"
    description: str      # one-liner for the picker
    doc_title: str        # what gets produced, e.g. "RTI application"
    icon: Optional[str] = None  # lucide icon name hint for the UI
    fields: list[FieldSpec] = Field(default_factory=list)
    note: Optional[str] = None  # an honesty note shown in the form (e.g. "fees vary by state")


class DraftSection(BaseModel):
    """A named block of procedural guidance, e.g. 'How to file' / 'If you're ignored'."""

    label: str
    items: list[str] = Field(default_factory=list)
    tone: SectionTone = "default"


class DraftRequest(BaseModel):
    journey: str = Field(..., max_length=64)
    # Free-form per-journey inputs, keyed by FieldSpec.key. Values are strings or booleans.
    fields: dict[str, Union[str, bool]] = Field(default_factory=dict)
    # Shared optional applicant details for the signature block.
    applicant_name: Optional[str] = Field(default=None, max_length=120)
    applicant_address: Optional[str] = Field(default=None, max_length=500)
    language: str = Field(default="en", max_length=8)


class DraftResponse(BaseModel):
    journey: str
    doc_title: str
    # The ready-to-use document (the main deliverable).
    document_text: str
    subject_line: Optional[str] = None
    # The framed content as a list (e.g. RTI questions, complaint grievances) — a summary.
    key_points: list[str] = Field(default_factory=list)
    # Procedural scaffolding: where/how to file, timeline, escalation, tips — journey-defined.
    sections: list[DraftSection] = Field(default_factory=list)

    # --- Trust contract ---
    confidence: Confidence
    citations: list[Citation] = Field(default_factory=list)
    citation_verified: bool = True
    disclaimer: str = DRAFT_DISCLAIMER
    response_time_ms: int
