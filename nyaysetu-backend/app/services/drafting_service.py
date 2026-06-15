"""Generic drafting engine: dispatches to a registered Journey.

The same orchestration serves every document type. The journey owns what's specific
(fields, framing prompt, template, citations, procedural sections); this service runs
the common flow and keeps the LLM on a short leash:

    validate required inputs
      -> if the journey has a framing prompt, ask the LLM to turn the situation into the
         document's content; parse it (journey-specific), else fall back to raw inputs
      -> render the document deterministically
      -> attach the journey's hand-authored citations + sections

The LLM can only ever affect the *content wording*; legal facts come from the journey.
"""
from __future__ import annotations

import time
from typing import Optional

from app.knowledge.drafting import all_journeys, get_journey
from app.knowledge.drafting.base import Journey
from app.schemas.draft import DRAFT_DISCLAIMER, DraftResponse, JourneyInfo
from app.services.llm_service import LLMError, OllamaClient, get_llm
from app.utils.logger import get_logger

logger = get_logger(__name__)


class JourneyNotFound(ValueError):
    pass


class MissingFields(ValueError):
    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        super().__init__(f"Missing required field(s): {', '.join(missing)}")


def list_journeys() -> list[JourneyInfo]:
    """Public catalogue for the frontend to build its picker + forms generically."""
    return [
        JourneyInfo(
            id=j.id,
            title=j.title,
            description=j.description,
            doc_title=j.doc_title,
            icon=j.icon,
            note=j.note,
            fields=j.fields(),
        )
        for j in all_journeys()
    ]


class DraftingService:
    def __init__(self, llm: OllamaClient | None = None) -> None:
        self._llm = llm or get_llm()

    def draft(
        self,
        *,
        journey_id: str,
        fields: dict,
        applicant_name: Optional[str] = None,
        applicant_address: Optional[str] = None,
        language: str = "en",
    ) -> DraftResponse:
        started = time.perf_counter()
        journey = get_journey(journey_id)
        if journey is None:
            raise JourneyNotFound(journey_id)

        inputs = self._normalize(journey, fields)
        self._require(journey, inputs)

        # 1. Frame the situation into the document's content (the LLM's only job).
        framed, llm_ok = self._frame(journey, inputs)
        if not framed:
            framed = journey.fallback_framed(inputs)

        # 2. Render + scaffold — all deterministic, from the journey.
        result = journey.render(
            inputs=inputs,
            framed=framed,
            applicant_name=(applicant_name or "").strip(),
            applicant_address=(applicant_address or "").strip(),
        )
        citations = journey.citations()
        sections = journey.sections(inputs)
        confidence = journey.confidence(inputs=inputs, framed=framed, llm_ok=llm_ok)

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "Drafted %r in %dms | confidence=%s | llm_ok=%s | points=%d",
            journey_id, elapsed_ms, confidence, llm_ok, len(result.key_points),
        )

        return DraftResponse(
            journey=journey.id,
            doc_title=journey.doc_title,
            document_text=result.document_text,
            subject_line=result.subject_line,
            key_points=result.key_points,
            sections=sections,
            confidence=confidence,
            citations=citations,
            citation_verified=True,  # citations come from the journey's hand-authored facts
            disclaimer=DRAFT_DISCLAIMER,
            response_time_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize(journey: Journey, fields: dict) -> dict:
        """Keep only declared fields; coerce checkbox values to bool, others to str."""
        out: dict = {}
        for spec in journey.fields():
            if spec.key not in fields:
                continue
            val = fields[spec.key]
            if spec.kind == "checkbox":
                out[spec.key] = bool(val)
            else:
                out[spec.key] = str(val).strip()
        return out

    @staticmethod
    def _require(journey: Journey, inputs: dict) -> None:
        missing = [
            s.label for s in journey.fields()
            if s.required and not str(inputs.get(s.key, "")).strip()
        ]
        if missing:
            raise MissingFields(missing)

    def _frame(self, journey: Journey, inputs: dict) -> tuple[dict, bool]:
        """Returns (framed_content, llm_ok). No prompt => deterministic journey (ok=True)."""
        prompt = journey.framing_prompt(inputs)
        if prompt is None:
            return journey.fallback_framed(inputs), True
        try:
            raw = self._llm.generate_json(prompt)
        except LLMError as e:
            logger.warning("Drafting framing LLM failed for %r (%s) — using fallback.", journey.id, e)
            return {}, False
        framed = journey.parse_framed(raw, inputs)
        return (framed, True) if framed else ({}, False)
