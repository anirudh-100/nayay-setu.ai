"""Journey: draft a written police complaint requesting registration of an FIR.

For a cognizable offence, the police MUST register an FIR (BNSS s.173, the successor to
CrPC s.154). This journey turns the citizen's account into a clear, factual written
complaint to the Station House Officer, and lays out the escalation ladder if the police
refuse (SP under s.173(4); Magistrate under s.175(3)).

Trust note: the BNSS sections cited are from the official India Code BNSS already in the
corpus; the snippets here are concise hand-authored paraphrases, so they're marked
``curated`` with the BNSS as the authority. The LLM only turns the account into factual
statements — it does NOT assign offence sections (the police/Magistrate determine those).
"""
from __future__ import annotations

from typing import Optional
from urllib.parse import quote_plus

from app.knowledge.drafting.base import Journey, RenderResult, coerce_lines, register
from app.rag.models import Citation
from app.schemas.draft import DraftSection, FieldSpec

EMERGENCY = "112"
_AUTHORITY = "Bharatiya Nagarik Suraksha Sanhita, 2023 (official India Code; section summaries curated)"


def _cite(section: str, snippet: str) -> Citation:
    return Citation(
        label=f"BNSS Section {section}",
        source_type="statute",
        snippet=snippet,
        url=f"https://indiankanoon.org/search/?formInput={quote_plus(f'section {section} bharatiya nagarik suraksha sanhita')}",
        code_status="current",
        verification="curated",
        source_authority=_AUTHORITY,
    )


_PROMPT = """You are helping an Indian citizen write a police complaint to register an FIR.

What happened (in their words): "{incident}"
{accused_block}
Turn this into clear, factual statements for the complaint.

Rules:
- "facts": 2 to 6 STRINGS, each a single factual sentence describing what happened, in order. Include date/time/place where the person gave them; use a blank like __________ where a detail is missing. State only facts, never legal conclusions or offence sections. Never return a number.
- Use calm, neutral, first-person language ("On __________, I ...").

Here is an EXAMPLE of the exact JSON format and style to return:
{{
  "facts": [
    "On __________ at about __________, near __________, two unidentified men on a motorcycle snatched my mobile phone and purse.",
    "They threatened me and fled towards __________; the incident was witnessed by __________.",
    "The stolen items include __________ worth approximately ₹__________."
  ]
}}

Now return ONLY the JSON object for THIS incident (no prose before or after):
"""


class PoliceComplaintJourney(Journey):
    id = "police_complaint"
    title = "Police complaint (FIR)"
    description = "Report a crime and ask the police to register an FIR."
    doc_title = "Police complaint"
    icon = "shield"
    note = "If you are in immediate danger, call 112 first. This drafts a written complaint to take or post to the police station."

    def fields(self) -> list[FieldSpec]:
        return [
            FieldSpec(
                key="incident",
                label="What happened?",
                kind="textarea",
                required=True,
                placeholder="Describe the incident: when, where, what was done, and by whom (if known).",
            ),
            FieldSpec(
                key="accused",
                label="Who is involved? (if known)",
                placeholder="Names or descriptions of the persons involved",
            ),
            FieldSpec(
                key="station",
                label="Police station (if you know it)",
                placeholder="e.g. Shivajinagar Police Station, Pune",
            ),
        ]

    # --- framing ---
    def framing_prompt(self, inputs: dict) -> Optional[str]:
        accused = str(inputs.get("accused") or "").strip()
        block = f'Persons involved: "{accused}"\n' if accused else ""
        return _PROMPT.format(incident=str(inputs.get("incident", "")).strip(), accused_block=block)

    def parse_framed(self, raw: dict, inputs: dict) -> dict:
        facts = coerce_lines(raw.get("facts"))
        return {"facts": facts} if facts else {}

    def fallback_framed(self, inputs: dict) -> dict:
        incident = " ".join(str(inputs.get("incident", "")).split())
        return {"facts": [incident] if incident else ["[Describe what happened, with date, time and place]"]}

    # --- render ---
    def render(self, *, inputs, framed, applicant_name, applicant_address) -> RenderResult:
        name = applicant_name or "[Your full name]"
        address = applicant_address or "[Your full postal address]"
        station = str(inputs.get("station") or "").strip() or "______________ Police Station"
        accused = str(inputs.get("accused") or "").strip()
        facts: list[str] = framed.get("facts") or ["[Describe what happened, with date, time and place]"]
        numbered = "\n".join(f"{i}. {f}" for i, f in enumerate(facts, start=1))
        accused_para = (
            f"\nThe person(s) involved: {accused}.\n" if accused else ""
        )

        text = (
            "To,\n"
            "The Station House Officer (SHO),\n"
            f"{station}\n\n"
            "Subject: Complaint and request to register an FIR\n\n"
            "Respected Sir/Madam,\n\n"
            f"I, {name}, resident of {address}, wish to report the following incident and request that "
            "a First Information Report (FIR) be registered:\n\n"
            f"{numbered}\n"
            f"{accused_para}\n"
            "The above acts appear to constitute a cognizable offence. I request you to kindly register "
            "an FIR under Section 173 of the Bharatiya Nagarik Suraksha Sanhita, 2023, investigate the "
            "matter, and take appropriate action in accordance with law. I am ready to provide any "
            "further information, statement, or evidence required.\n\n"
            "Kindly provide me a free copy of the registered FIR as required by law.\n\n"
            "Yours faithfully,\n\n"
            f"{name}\n"
            f"{address}\n"
            "Phone: __________\n\n"
            "Date: __________          Place: __________"
        )
        return RenderResult(document_text=text, subject_line="Request to register an FIR", key_points=facts)

    # --- scaffolding ---
    def citations(self) -> list[Citation]:
        return [
            _cite("173", "Information about a cognizable offence may be given to the police orally or "
                         "electronically; it must be recorded, read over to the informant, and a free copy "
                         "given. (Successor to CrPC Section 154 — the FIR provision.)"),
            _cite("173(4)", "If the officer in charge refuses to record the information, the aggrieved person "
                            "may send it in writing by post to the Superintendent of Police."),
            _cite("175(3)", "A Magistrate may order an investigation into a cognizable offence, including where "
                            "the police have not registered or acted on the complaint."),
            _cite("174", "Information about a non-cognizable offence is entered in a register and the informant "
                         "is referred to the Magistrate; police do not investigate without the Magistrate's order."),
        ]

    def sections(self, inputs: dict) -> list[DraftSection]:
        return [
            DraftSection(
                label="How & where to file",
                tone="info",
                items=[
                    "Take or post the signed complaint to the Station House Officer (SHO). For a cognizable "
                    "offence the police MUST register an FIR (BNSS s.173).",
                    "Zero FIR: you can report at ANY police station regardless of where the offence happened — "
                    "it is then transferred to the correct station.",
                    "You are entitled to a FREE copy of the registered FIR. Keep it safe and note the FIR number.",
                ],
            ),
            DraftSection(
                label="If the police refuse to register it",
                tone="warn",
                items=[
                    "Send the written complaint by registered post to the Superintendent of Police (SP) "
                    "(BNSS s.173(4)). Keep the posting receipt.",
                    "If there is still no action, file a complaint before the Magistrate to direct an "
                    "investigation (BNSS s.175(3)).",
                ],
            ),
            DraftSection(
                label="Good to know",
                items=[
                    "If you are in immediate danger or it is an emergency, call 112 first.",
                    "For a non-cognizable (minor) matter, the police record it and refer you to the Magistrate "
                    "(BNSS s.174).",
                    "Include date, time, place, exactly what happened, descriptions of those involved, and any "
                    "witnesses or evidence. Attach copies of any documents.",
                ],
            ),
        ]


register(PoliceComplaintJourney())
