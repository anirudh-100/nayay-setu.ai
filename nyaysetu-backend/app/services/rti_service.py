"""RTI drafting: turn a citizen's plain-language request into a ready-to-file RTI.

The flow keeps the language model on a very short leash, because this is a trust-critical
drafting feature:

    citizen's request ("I want to know the status of my passport application")
      -> LLM rephrases it into specific, answerable RTI questions + a subject line
         (the ONLY job the model has here)
      -> the application letter is assembled from a fixed template (deterministic)
      -> the legal scaffolding — governing sections, fee, time limits, appeals — is
         pulled verbatim from app.knowledge.rti (hand-authored from the official Act)

So every legal fact is correct by construction, and the model can only ever affect the
*wording of the questions*, never the law. If the model fails, we fall back to the user's
own text as a single question and still produce a valid, filable application.
"""
from __future__ import annotations

import re
import time
from typing import Optional

from app.knowledge import rti as rti_kb
from app.schemas.rti import FilingInfo, RTIDraftResponse
from app.schemas.ask import Confidence
from app.services.llm_service import LLMError, OllamaClient
from app.utils.logger import get_logger

logger = get_logger(__name__)

_MAX_QUESTIONS = 8
_AUTHORITY_PLACEHOLDER = "[Name of the public authority / department that holds this information]"
_NAME_PLACEHOLDER = "[Your full name]"
_ADDRESS_PLACEHOLDER = "[Your full postal address]"

PROMPT_TEMPLATE = """You are helping an Indian citizen file a Right to Information (RTI) application.

The citizen wants to know:
"{subject}"
{authority_block}
Rewrite this as specific Right to Information questions a Public Information Officer must answer from government records.

Rules:
- Each question is a STRING — a full, polite sentence asking for information, records, documents, file notings, or a status update that would exist on record. Never return a number.
- Do NOT ask for opinions, reasons, justifications, or "why" questions — a PIO is not obliged to answer those.
- Ask for certified copies of the relevant documents where useful.
- Give 2 to 5 questions, each a single clear ask.

Here is an EXAMPLE of the exact JSON format and style to return:
{{
  "subject_line": "Status of ration card application",
  "questions": [
    "Please provide the current status of my ration card application dated __________.",
    "Please provide certified copies of all file notings and correspondence on this application.",
    "Please state the name and designation of the officer currently handling this application."
  ]
}}

Now return ONLY the JSON object for THIS citizen's request (no prose before or after):
"""


# Strip a leading list marker ("1. ", "2) ", "- ", "• ") without eating a leading year.
_LEADING_MARKER = re.compile(r"^\s*(?:\d{1,2}[.)]|[-•*])\s+")


def _coerce_questions(value: object) -> list[str]:
    """Coerce the LLM 'questions' field into clean question strings.

    Small models sometimes emit a *numeric* array (e.g. [1.0, 2.2, ...]) when told to
    "number" the questions — so we drop any item with no alphabetic content rather than
    letting a bare number masquerade as a question.
    """
    raw = value if isinstance(value, (list, tuple)) else ([value] if isinstance(value, str) else [])
    items: list[str] = []
    for item in raw:
        if item is None:
            continue
        s = _LEADING_MARKER.sub("", str(item).strip()).strip()
        if s and any(ch.isalpha() for ch in s):  # must contain real words, not just a number
            items.append(s)
    return items[:_MAX_QUESTIONS]


class RTIService:
    def __init__(self, llm: OllamaClient | None = None) -> None:
        self._llm = llm or OllamaClient()

    def draft(
        self,
        *,
        subject: str,
        public_authority: Optional[str] = None,
        level: str = "central",
        applicant_name: Optional[str] = None,
        applicant_address: Optional[str] = None,
        is_bpl: bool = False,
        language: str = "en",
    ) -> RTIDraftResponse:
        started = time.perf_counter()
        subject = subject.strip()
        authority = (public_authority or "").strip()

        # 1. LLM: frame the request into specific questions (its only job). Degrade
        #    gracefully to the raw request if the model is unavailable or returns nothing.
        questions, subject_line, llm_ok = self._frame_questions(subject, authority)
        if not questions:
            questions = [subject]
            subject_line = subject_line or self._fallback_subject(subject)

        # 2. Deterministic assembly — letter + all legal scaffolding from the knowledge base.
        authority_display = authority or _AUTHORITY_PLACEHOLDER
        application_text = self._render_application(
            authority=authority_display,
            subject_line=subject_line,
            questions=questions,
            applicant_name=(applicant_name or "").strip() or _NAME_PLACEHOLDER,
            applicant_address=(applicant_address or "").strip() or _ADDRESS_PLACEHOLDER,
            is_bpl=is_bpl,
        )

        filing = rti_kb.filing_guidance(level=level, is_bpl=is_bpl)
        citations = rti_kb.core_citations()

        # Confidence reflects how complete the inputs were — never overstated. The legal
        # facts are always correct, so this is about the *draft's* readiness to send.
        if llm_ok and authority:
            confidence: Confidence = "high"
        elif llm_ok or authority:
            confidence = "medium"
        else:
            confidence = "low"

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "Drafted RTI in %dms | level=%s | bpl=%s | questions=%d | authority=%s | llm_ok=%s",
            elapsed_ms,
            level,
            is_bpl,
            len(questions),
            "yes" if authority else "placeholder",
            llm_ok,
        )

        return RTIDraftResponse(
            application_text=application_text,
            subject_line=subject_line,
            questions=questions,
            public_authority=authority_display,
            filing=FilingInfo(**filing),
            timeline=rti_kb.timeline_guidance(),
            appeals=rti_kb.appeal_guidance(),
            tips=rti_kb.drafting_tips(),
            confidence=confidence,
            citations=citations,
            citation_verified=True,  # citations are curated facts from app.knowledge.rti
            response_time_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------ #
    def _frame_questions(self, subject: str, authority: str) -> tuple[list[str], str, bool]:
        """Ask the LLM to turn the request into RTI questions. Returns (questions, subject_line, ok)."""
        authority_block = f'The information is held by: "{authority}"\n' if authority else ""
        prompt = PROMPT_TEMPLATE.format(subject=subject, authority_block=authority_block)
        try:
            raw = self._llm.generate_json(prompt)
        except LLMError as e:
            logger.warning("RTI question-framing LLM failed (%s) — using raw request.", e)
            return [], "", False

        questions = _coerce_questions(raw.get("questions"))
        subject_line = str(raw.get("subject_line") or "").strip()[:120]
        return questions, subject_line, bool(questions)

    @staticmethod
    def _fallback_subject(subject: str) -> str:
        s = " ".join(subject.split())
        return (s[:60] + "…") if len(s) > 60 else s

    @staticmethod
    def _render_application(
        *,
        authority: str,
        subject_line: str,
        questions: list[str],
        applicant_name: str,
        applicant_address: str,
        is_bpl: bool,
    ) -> str:
        numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, start=1))
        if is_bpl:
            fee_para = (
                "I belong to the Below Poverty Line (BPL) category and am therefore exempt "
                "from the application fee under Section 7(5) of the Act. A copy of my BPL "
                "certificate is enclosed."
            )
        else:
            fee_para = (
                "The prescribed application fee of ₹10 is enclosed by way of Indian Postal "
                "Order / Demand Draft No. __________ dated __________ (please confirm the fee "
                "and accepted payment mode for the concerned authority)."
            )

        return (
            "To,\n"
            "The Public Information Officer (PIO),\n"
            f"{authority}\n\n"
            f"Subject: Request for information under the Right to Information Act, 2005 — {subject_line}\n\n"
            "Sir/Madam,\n\n"
            "Under Section 6(1) of the Right to Information Act, 2005, I request the following "
            "information:\n\n"
            f"{numbered}\n\n"
            f"{fee_para}\n\n"
            "If any part of the information sought is held by or is more closely connected with "
            "another public authority, kindly transfer this application to the appropriate "
            "authority under Section 6(3) of the Act and inform me of the same.\n\n"
            "Kindly provide the information within the period prescribed under Section 7 of the "
            "Act. I declare that I am a citizen of India.\n\n"
            "Yours faithfully,\n\n"
            f"{applicant_name}\n"
            f"{applicant_address}\n\n"
            "Date: __________          Place: __________"
        )
