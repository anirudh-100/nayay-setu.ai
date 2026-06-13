"""Journey: draft a Right to Information application.

Wraps the source-attributed facts in ``app.knowledge.rti`` (sections, fee, time limits,
appeals — hand-authored from the official RTI Act, 2005) as a drafting Journey. The LLM
only turns the citizen's request into specific, answerable RTI questions; the letter and
every legal fact are assembled deterministically.
"""
from __future__ import annotations

from typing import Optional

from app.knowledge import rti as kb
from app.knowledge.drafting.base import Journey, RenderResult, coerce_lines, register
from app.rag.models import Citation
from app.schemas.ask import Confidence
from app.schemas.draft import DraftSection, FieldOption, FieldSpec

_AUTHORITY_PLACEHOLDER = "[Name of the public authority / department that holds this information]"
_NAME_PLACEHOLDER = "[Your full name]"
_ADDRESS_PLACEHOLDER = "[Your full postal address]"

_PROMPT = """You are helping an Indian citizen file a Right to Information (RTI) application.

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


class RTIJourney(Journey):
    id = "rti"
    title = "Demand information (RTI)"
    description = "Ask a government office for information they must legally provide."
    doc_title = "RTI application"
    icon = "scroll-text"
    note = "Central-government defaults shown; State fees and portals vary."

    def fields(self) -> list[FieldSpec]:
        return [
            FieldSpec(
                key="subject",
                label="What do you want to know?",
                kind="textarea",
                required=True,
                placeholder="e.g. The current status of my passport application and why it is delayed",
            ),
            FieldSpec(
                key="public_authority",
                label="Department / office (if you know)",
                placeholder="e.g. Regional Passport Office, Pune",
            ),
            FieldSpec(
                key="level",
                label="Government level",
                kind="select",
                default="central",
                options=[
                    FieldOption(value="central", label="Central Government"),
                    FieldOption(value="state", label="State Government"),
                ],
            ),
            FieldSpec(
                key="is_bpl",
                label="I have a Below Poverty Line (BPL) card — I'm exempt from the fee",
                kind="checkbox",
            ),
        ]

    # --- framing ---
    def framing_prompt(self, inputs: dict) -> Optional[str]:
        authority = str(inputs.get("public_authority") or "").strip()
        block = f'The information is held by: "{authority}"\n' if authority else ""
        return _PROMPT.format(subject=str(inputs.get("subject", "")).strip(), authority_block=block)

    def parse_framed(self, raw: dict, inputs: dict) -> dict:
        questions = coerce_lines(raw.get("questions"))
        if not questions:
            return {}
        return {"questions": questions, "subject_line": str(raw.get("subject_line") or "").strip()[:120]}

    def fallback_framed(self, inputs: dict) -> dict:
        subject = " ".join(str(inputs.get("subject", "")).split())
        return {"questions": [subject], "subject_line": (subject[:60] + "…") if len(subject) > 60 else subject}

    # --- render ---
    def render(self, *, inputs, framed, applicant_name, applicant_address) -> RenderResult:
        questions: list[str] = framed.get("questions") or [str(inputs.get("subject", ""))]
        subject_line = framed.get("subject_line") or "Request for information"
        authority = str(inputs.get("public_authority") or "").strip() or _AUTHORITY_PLACEHOLDER
        is_bpl = bool(inputs.get("is_bpl"))
        numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, start=1))

        if is_bpl:
            fee_para = (
                "I belong to the Below Poverty Line (BPL) category and am therefore exempt from "
                "the application fee under Section 7(5) of the Act. A copy of my BPL certificate "
                "is enclosed."
            )
        else:
            fee_para = (
                "The prescribed application fee of ₹10 is enclosed by way of Indian Postal Order / "
                "Demand Draft No. __________ dated __________ (please confirm the fee and accepted "
                "payment mode for the concerned authority)."
            )

        text = (
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
            "another public authority, kindly transfer this application to the appropriate authority "
            "under Section 6(3) of the Act and inform me of the same.\n\n"
            "Kindly provide the information within the period prescribed under Section 7 of the Act. "
            "I declare that I am a citizen of India.\n\n"
            "Yours faithfully,\n\n"
            f"{applicant_name or _NAME_PLACEHOLDER}\n"
            f"{applicant_address or _ADDRESS_PLACEHOLDER}\n\n"
            "Date: __________          Place: __________"
        )
        return RenderResult(document_text=text, subject_line=subject_line, key_points=questions)

    # --- scaffolding ---
    def citations(self) -> list[Citation]:
        return kb.core_citations()

    def sections(self, inputs: dict) -> list[DraftSection]:
        level = str(inputs.get("level") or "central")
        is_bpl = bool(inputs.get("is_bpl"))
        filing = kb.filing_guidance(level=level, is_bpl=is_bpl)
        file_items = [filing["where_to_file"], filing["fee"]]
        if filing.get("portal"):
            file_items.append(f"Online portal: {filing['portal']}")
        return [
            DraftSection(label="How & where to file", items=file_items, tone="info"),
            DraftSection(label="When to expect a reply", items=kb.timeline_guidance()),
            DraftSection(label="If you don't get a reply", items=kb.appeal_guidance(), tone="warn"),
            DraftSection(label="Tips for a useful reply", items=kb.drafting_tips()),
        ]

    def confidence(self, *, inputs, framed, llm_ok) -> Confidence:
        authority = bool(str(inputs.get("public_authority") or "").strip())
        if llm_ok and authority:
            return "high"
        if llm_ok or authority:
            return "medium"
        return "low"


register(RTIJourney())
