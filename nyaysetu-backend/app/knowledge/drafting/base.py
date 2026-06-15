"""The Journey abstraction + registry that the generic drafting engine dispatches to.

A Journey owns everything specific to one document type. The engine
(app.services.drafting_service) runs the same orchestration for every journey:

    validate inputs
      -> if the journey defines a framing prompt, ask the LLM to turn the situation
         into the document's specific content (questions, grievances, a narrative)
      -> parse that (journey-specific), or fall back to the raw inputs if the LLM is down
      -> render the document deterministically from a template
      -> attach the journey's citations + procedural sections (all hand-authored)

So the model only ever affects the *wording of the content*, never the legal facts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from app.rag.models import Citation
from app.schemas.ask import Confidence
from app.schemas.draft import DraftSection, FieldSpec


@dataclass
class RenderResult:
    document_text: str
    subject_line: Optional[str] = None
    key_points: list[str] = field(default_factory=list)


class Journey:
    """Base class for a drafting journey. Subclasses set the class attrs and override
    the hooks they need. Sensible defaults keep simple journeys terse."""

    id: str = ""
    title: str = ""
    description: str = ""
    doc_title: str = ""
    icon: str = "file-text"
    note: Optional[str] = None

    # --- declaration ---
    def fields(self) -> list[FieldSpec]:
        return []

    # --- LLM framing (optional) ---
    def framing_prompt(self, inputs: dict) -> Optional[str]:
        """Prompt to turn the situation into the document's content. None = no LLM step."""
        return None

    def parse_framed(self, raw: dict, inputs: dict) -> dict:
        """Coerce the LLM JSON into usable framed content. Return {} if unusable
        (the engine then falls back to ``fallback_framed``)."""
        return {}

    def fallback_framed(self, inputs: dict) -> dict:
        """Framed content derived purely from the raw inputs (no LLM)."""
        return {}

    # --- rendering ---
    def render(
        self, *, inputs: dict, framed: dict, applicant_name: str, applicant_address: str
    ) -> RenderResult:
        raise NotImplementedError

    # --- legal scaffolding (hand-authored, sourced) ---
    def citations(self) -> list[Citation]:
        return []

    def sections(self, inputs: dict) -> list[DraftSection]:
        return []

    # --- confidence (readiness of the draft, never overstated) ---
    def confidence(self, *, inputs: dict, framed: dict, llm_ok: bool) -> Confidence:
        if self.framing_prompt(inputs) is None:
            return "high"  # fully deterministic journey
        return "high" if llm_ok else "medium"


# --- shared helpers ------------------------------------------------------- #
_LEADING_MARKER = re.compile(r"^\s*(?:\d{1,2}[.)]|[-•*])\s+")


def coerce_lines(value: object, *, limit: int = 8) -> list[str]:
    """Coerce an LLM list field into clean text lines.

    Small models sometimes emit a numeric array when asked to "number" items, so we drop
    any entry with no alphabetic content rather than letting a bare number through.
    """
    raw = value if isinstance(value, (list, tuple)) else ([value] if isinstance(value, str) else [])
    out: list[str] = []
    for item in raw:
        if item is None:
            continue
        s = _LEADING_MARKER.sub("", str(item).strip()).strip()
        if s and any(ch.isalpha() for ch in s):
            out.append(s)
    return out[:limit]


# --- registry ------------------------------------------------------------- #
_REGISTRY: "dict[str, Journey]" = {}
_ORDER: list[str] = []


def register(journey: Journey) -> Journey:
    if not journey.id:
        raise ValueError("Journey must define a non-empty id")
    if journey.id not in _REGISTRY:
        _ORDER.append(journey.id)
    _REGISTRY[journey.id] = journey
    return journey


def get_journey(journey_id: str) -> Optional[Journey]:
    return _REGISTRY.get(journey_id)


def all_journeys() -> list[Journey]:
    return [_REGISTRY[i] for i in _ORDER]
