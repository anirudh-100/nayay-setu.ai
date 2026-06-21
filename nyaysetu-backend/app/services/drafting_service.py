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

import json
import re
import time
from typing import Optional

from app.knowledge.drafting import all_journeys, get_journey
from app.knowledge.drafting.base import Journey, RenderResult, coerce_lines
from app.schemas.draft import DRAFT_DISCLAIMER, DRAFT_DISCLAIMER_HI, DraftResponse, DraftSection, JourneyInfo
from app.services.llm_service import LLMError, OllamaClient, get_llm
from app.utils.logger import get_logger

logger = get_logger(__name__)

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def _wants_hindi(language: str) -> bool:
    return (language or "").strip().lower().startswith("hi")


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

        # Hindi: render English deterministically (as above), then translate the draft +
        # notes into Hindi keeping every law/section reference, name, date and amount intact
        # — mirroring how /ask and /analyze answer in Hindi with citations kept standard.
        disclaimer = DRAFT_DISCLAIMER
        if _wants_hindi(language):
            localized = self._localize_hindi(result, sections)
            if localized is not None:
                result, sections = localized
                disclaimer = DRAFT_DISCLAIMER_HI

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "Drafted %r in %dms | confidence=%s | llm_ok=%s | points=%d | hindi=%s",
            journey_id, elapsed_ms, confidence, llm_ok, len(result.key_points), _wants_hindi(language),
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
            disclaimer=disclaimer,
            response_time_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------ #
    def _localize_hindi(
        self, result: RenderResult, sections: list[DraftSection]
    ) -> Optional[tuple[RenderResult, list[DraftSection]]]:
        """Translate a rendered English draft into Hindi, preserving law/section references,
        names, addresses, dates, reference numbers and amounts verbatim. Returns the Hindi
        (RenderResult, sections), or None on any failure so the caller keeps the English
        draft (never a half-translated one)."""
        try:
            payload = {
                "document_text": result.document_text,
                "subject_line": result.subject_line or "",
                "key_points": list(result.key_points or []),
                "sections": [{"label": s.label, "items": list(s.items)} for s in sections],
            }
            prompt = (
                "Translate this Indian legal draft and its notes into simple, formal Hindi that "
                "an ordinary person can read and file. STRICT RULES:\n"
                "- Keep ALL law/act names and section numbers EXACTLY as-is (e.g. \"RTI Act, 2005\", "
                "\"Section 6(1)\", \"BNS Section 318\", \"Article 21\"). Do NOT translate or transliterate them.\n"
                "- Keep proper nouns, person/office names, addresses, dates, reference numbers and money "
                "amounts (e.g. Rs. 5,000) EXACTLY as-is.\n"
                "- Preserve the document's structure, line breaks (\\n) and any [blanks]/placeholders.\n"
                "- Translate only the surrounding prose; do not add, remove, or reorder content.\n"
                "Return ONLY JSON with the SAME keys and shape as the input:\n"
                '{"document_text":"...","subject_line":"...","key_points":["..."],'
                '"sections":[{"label":"...","items":["..."]}]}\n\nINPUT:\n'
                + json.dumps(payload, ensure_ascii=False)
            )
            raw = self._llm.generate_json(prompt)

            document_text = str(raw.get("document_text") or "").strip()
            # Guard: if the model didn't actually return Hindi, keep the English draft.
            if not document_text or not _DEVANAGARI.search(document_text):
                return None
            subject_line = str(raw.get("subject_line") or "").strip() or result.subject_line
            key_points = coerce_lines(raw.get("key_points")) or list(result.key_points or [])

            new_sections = sections
            raw_secs = raw.get("sections")
            if isinstance(raw_secs, list) and len(raw_secs) == len(sections):
                new_sections = []
                for orig, rs in zip(sections, raw_secs):
                    if isinstance(rs, dict):
                        label = str(rs.get("label") or "").strip() or orig.label
                        items = coerce_lines(rs.get("items")) or list(orig.items)
                        new_sections.append(DraftSection(label=label, items=items, tone=orig.tone))
                    else:
                        new_sections.append(orig)

            return RenderResult(
                document_text=document_text, subject_line=subject_line, key_points=key_points
            ), new_sections
        except Exception as e:  # never let translation break a draft — keep the English one
            logger.warning("Hindi draft localization skipped (%s) — returning English draft.", e)
            return None

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
