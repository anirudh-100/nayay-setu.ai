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

# Hindi versions of the trust strings, surfaced when the user asks in Hindi. The
# disclaimer especially must be readable in the user's language — a Hindi-only reader
# must understand this is information, not advice.
DISCLAIMER_HI = (
    "यह केवल सामान्य जानकारी के लिए है, कानूनी सलाह नहीं। अपनी विशेष स्थिति के लिए "
    "किसी योग्य वकील से सलाह ज़रूर लें।"
)
LEGAL_AID_ESCALATION_HI = (
    "मुफ़्त, आधिकारिक कानूनी सहायता के लिए अपने ज़िला विधिक सेवा प्राधिकरण (DLSA) से "
    "संपर्क करें या नालसा (NALSA) हेल्पलाइन 15100 पर कॉल करें।"
)


class ConversationTurn(BaseModel):
    """One prior turn, sent so a follow-up question can be resolved into a standalone one."""
    role: str = Field(default="", max_length=16)       # "user" | "assistant"
    content: str = Field(default="", max_length=4000)


class CaseAnalysis(BaseModel):
    """A structured, citizen-ordered breakdown of a described situation.

    Built ONLY on a strong, citation-verified (Mode A / high-confidence) answer, and only
    after deterministic server-side scrubbing keeps every claim inside the trust contract:
    no outcome/verdict prediction, no ungrounded offence classification
    (cognizable/bailable/…), and no invented case-law citations. Any weakness collapses
    the whole block back to None — the citizen then sees today's honest single paragraph,
    never an invented six-section skeleton. Empty lists render as hidden sections.
    """
    # Set in code (never from the LLM) so the "not a prediction" frame can't be dropped.
    outcome_framing: str = ""
    situation: list[str] = Field(default_factory=list)          # what the facts legally are
    applicable_law: list[str] = Field(default_factory=list)     # cited sections (current-first)
    what_happens_next: list[str] = Field(default_factory=list)  # BNSS process steps (grounded)
    do_now: list[str] = Field(default_factory=list)             # calm, concrete next steps
    also_possible: list[str] = Field(default_factory=list)      # impersonal possibilities
    for_your_advocate: list[str] = Field(default_factory=list)  # research pointers (no precedents)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    language: str = Field(default="en", max_length=8)
    # Recent conversation so follow-ups ("what's the punishment for that?") can be
    # resolved into a standalone question before retrieval. Used only to disambiguate —
    # never as a source of legal facts. Capped to keep prompts small.
    history: list[ConversationTurn] = Field(default_factory=list, max_length=12)


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
    # Optional structured case-analysis (situation → law → process → options → advocate
    # notes). Present only on a strong, verified answer; None means "show the plain card".
    analysis: Optional[CaseAnalysis] = None

    disclaimer: str = DISCLAIMER
    response_time_ms: int
