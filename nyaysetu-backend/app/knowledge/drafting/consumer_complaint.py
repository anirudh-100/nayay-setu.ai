"""Journey: draft a consumer complaint under the Consumer Protection Act, 2019.

For defective goods, deficient services, overcharging, or unfair trade practices. The
legal scaffolding (forum by claim value, two-year limitation, reliefs, e-Daakhil portal)
is hand-authored from the Consumer Protection Act, 2019 and its 2020/2021 Rules — marked
``curated`` with the source. The LLM only turns the citizen's story into clear grievance
points and the relief sought; the complaint is assembled deterministically.

Honesty notes: pecuniary limits are the post-2021 figures (District ≤ ₹50 lakh, State ≤
₹2 crore, National > ₹2 crore); fees are slab-based and small claims are free, but exact
fees and the bench's current limits should be confirmed before filing.
"""
from __future__ import annotations

from typing import Optional
from urllib.parse import quote_plus

from app.knowledge.drafting.base import Journey, RenderResult, coerce_lines, register
from app.rag.models import Citation
from app.schemas.draft import DraftSection, FieldSpec

E_DAAKHIL = "https://edaakhil.nic.in"
LIMITATION_YEARS = 2
_AUTHORITY = "Consumer Protection Act, 2019 (compiled from the official Act, India Code)"


def _cite(section: str, snippet: str) -> Citation:
    return Citation(
        label=f"Consumer Protection Act Section {section}",
        source_type="statute",
        snippet=snippet,
        url=f"https://indiankanoon.org/search/?formInput={quote_plus(f'section {section} consumer protection act 2019')}",
        code_status="current",
        verification="curated",
        source_authority=_AUTHORITY,
    )


_PROMPT = """You are helping an Indian citizen draft a consumer complaint under the Consumer Protection Act, 2019.

They are complaining against: "{opposite_party}"
What went wrong (in their words): "{grievance}"
{relief_block}
Turn this into the factual basis of a consumer complaint.

Rules:
- "grievance_points": 2 to 6 STRINGS, each a clear factual statement of what the business did wrong (defect in goods, deficiency in service, overcharging, or unfair trade practice). State facts, not opinions. Never return a number.
- "reliefs": 1 to 4 STRINGS, each a specific remedy the person wants (e.g. refund of the amount paid, replacement of the product, compensation for loss, removal of the defect).
- Keep each item one clear sentence. Use neutral, factual language.

Here is an EXAMPLE of the exact JSON format and style to return:
{{
  "grievance_points": [
    "The Complainant purchased a washing machine from the Opposite Party on __________ for ₹__________.",
    "The machine stopped working within two weeks and the Opposite Party refused to repair or replace it despite repeated requests."
  ],
  "reliefs": [
    "Direct the Opposite Party to replace the defective machine or refund the full amount of ₹__________.",
    "Direct the Opposite Party to pay compensation for the harassment and inconvenience caused."
  ]
}}

Now return ONLY the JSON object for THIS complaint (no prose before or after):
"""


class ConsumerComplaintJourney(Journey):
    id = "consumer_complaint"
    title = "Consumer complaint"
    description = "Complain about a defective product, bad service, or overcharging."
    doc_title = "Consumer complaint"
    icon = "shopping-bag"
    note = "Forum and fee depend on the amount involved; confirm your District Commission's current limits before filing."

    def fields(self) -> list[FieldSpec]:
        return [
            FieldSpec(
                key="opposite_party",
                label="Who are you complaining against?",
                required=True,
                placeholder="Business / company / seller name and address",
            ),
            FieldSpec(
                key="grievance",
                label="What went wrong?",
                kind="textarea",
                required=True,
                placeholder="e.g. I paid ₹15,000 for a phone that stopped working in a week and the shop refuses to refund or replace it.",
            ),
            FieldSpec(
                key="amount",
                label="Amount paid / value involved (₹)",
                placeholder="e.g. 15000",
                help="Decides which Commission hears it: up to ₹50 lakh — District.",
            ),
            FieldSpec(
                key="relief",
                label="What outcome do you want? (optional)",
                kind="textarea",
                placeholder="e.g. Full refund and compensation for the inconvenience",
            ),
        ]

    # --- framing ---
    def framing_prompt(self, inputs: dict) -> Optional[str]:
        relief = str(inputs.get("relief") or "").strip()
        relief_block = f'What they want: "{relief}"\n' if relief else ""
        return _PROMPT.format(
            opposite_party=str(inputs.get("opposite_party", "")).strip(),
            grievance=str(inputs.get("grievance", "")).strip(),
            relief_block=relief_block,
        )

    def parse_framed(self, raw: dict, inputs: dict) -> dict:
        points = coerce_lines(raw.get("grievance_points"))
        if not points:
            return {}
        reliefs = coerce_lines(raw.get("reliefs"))
        if not reliefs:
            reliefs = self._fallback_reliefs(inputs)
        return {"grievance_points": points, "reliefs": reliefs}

    def fallback_framed(self, inputs: dict) -> dict:
        grievance = " ".join(str(inputs.get("grievance", "")).split())
        return {"grievance_points": [grievance] if grievance else ["[Describe what went wrong]"],
                "reliefs": self._fallback_reliefs(inputs)}

    @staticmethod
    def _fallback_reliefs(inputs: dict) -> list[str]:
        relief = " ".join(str(inputs.get("relief") or "").split())
        if relief:
            return [relief]
        return ["Direct the Opposite Party to refund the amount paid and/or replace the goods, and "
                "pay compensation for the loss and inconvenience caused."]

    # --- render ---
    def render(self, *, inputs, framed, applicant_name, applicant_address) -> RenderResult:
        op = str(inputs.get("opposite_party") or "").strip() or "[Opposite Party — name and address]"
        name = applicant_name or "[Your full name]"
        address = applicant_address or "[Your full postal address]"
        points: list[str] = framed.get("grievance_points") or ["[Describe what went wrong]"]
        reliefs: list[str] = framed.get("reliefs") or []

        # Facts start at para 2 (para 1 establishes "consumer" status).
        facts = "\n".join(f"{i}. {p}" for i, p in enumerate(points, start=2))
        n = len(points) + 2  # next paragraph number after the facts
        relief_lines = "\n".join(
            f"   {chr(97 + i)}) {r}" for i, r in enumerate(reliefs)
        ) or "   a) Grant the Complainant appropriate relief as this Commission deems fit."

        text = (
            "BEFORE THE DISTRICT CONSUMER DISPUTES REDRESSAL COMMISSION, ______________ (District)\n\n"
            "Consumer Complaint No. __________ of 20____\n\n"
            "In the matter of:\n"
            f"{name}\n{address}\n"
            "                                                        … Complainant\n\n"
            "                          VERSUS\n\n"
            f"{op}\n"
            "                                                        … Opposite Party\n\n"
            "COMPLAINT UNDER SECTION 35 OF THE CONSUMER PROTECTION ACT, 2019\n\n"
            "The Complainant respectfully submits as follows:\n\n"
            "1. The Complainant is a \"consumer\" within the meaning of Section 2(7) of the Consumer "
            "Protection Act, 2019, having bought goods / availed services from the Opposite Party for "
            "consideration.\n\n"
            f"{facts}\n\n"
            f"{n}. The above acts of the Opposite Party amount to a defect in goods / deficiency in "
            "service / unfair trade practice under the Consumer Protection Act, 2019, and have caused "
            "the Complainant loss, hardship and mental agony.\n\n"
            f"{n + 1}. The cause of action arose on __________ at __________, which is within the "
            "territorial jurisdiction of this Commission. The complaint is filed within the limitation "
            "period of two years under Section 69 of the Act.\n\n"
            "PRAYER:\n"
            "The Complainant prays that this Commission may be pleased to direct the Opposite Party to:\n"
            f"{relief_lines}\n"
            "   and to pay the costs of this complaint, and pass any other order it deems fit and proper.\n\n"
            "VERIFICATION:\n"
            f"I, {name}, do hereby verify that the contents of this complaint are true and correct to "
            "the best of my knowledge and belief.\n\n"
            "Place: __________          Date: __________\n\n"
            f"                                                        {name}\n"
            "                                                        (Complainant)"
        )
        return RenderResult(document_text=text, subject_line="Consumer complaint", key_points=points)

    # --- scaffolding ---
    def citations(self) -> list[Citation]:
        return [
            _cite("2(7)", "Defines a \"consumer\" — a person who buys goods or avails services for "
                          "consideration (including online); excludes purchases for resale or commercial purpose."),
            _cite("34", "District Consumer Disputes Redressal Commission hears complaints where the value of "
                        "the goods or services paid does not exceed ₹50 lakh, and where the opposite party "
                        "resides/works, the cause of action arose, or the complainant resides or works."),
            _cite("35", "Manner of filing a complaint before the District Commission, including filing in "
                        "person or electronically and the fee payable."),
            _cite("39", "Reliefs the Commission may order — removal of defect, replacement, refund of price, "
                        "compensation for loss or injury, and discontinuance of unfair trade practice."),
            _cite("69", "A complaint must be filed within two years from the date on which the cause of action "
                        "arose (condonable for sufficient cause)."),
        ]

    def sections(self, inputs: dict) -> list[DraftSection]:
        return [
            DraftSection(
                label="Which forum & where to file",
                tone="info",
                items=[
                    "By value paid: up to ₹50 lakh — District Commission; ₹50 lakh to ₹2 crore — State "
                    "Commission; above ₹2 crore — National Commission (limits revised in 2021).",
                    "File where the opposite party resides or carries on business, where the cause of action "
                    "arose, OR where you reside or work (Consumer Protection Act, 2019).",
                    f"You can file online on the e-Daakhil portal: {E_DAAKHIL}",
                ],
            ),
            DraftSection(
                label="Fee & time limit",
                items=[
                    "Fee is nominal and slab-based by claim value; complaints up to ₹5 lakh carry no fee "
                    "(2020 Rules). Confirm the exact fee for your claim value.",
                    f"File within {LIMITATION_YEARS} years of the cause of action (Section 69).",
                ],
            ),
            DraftSection(
                label="Before you file (recommended)",
                tone="warn",
                items=[
                    "Send a written notice/demand to the business first, giving a short deadline to resolve — "
                    "it often works and strengthens your case if it doesn't.",
                    "Keep every bill, invoice, warranty card, screenshot, and message as evidence.",
                ],
            ),
            DraftSection(
                label="Tips",
                items=[
                    "Attach copies (not originals) of the bill/invoice, warranty, and all correspondence.",
                    "Quantify your loss clearly and state exactly what relief you want.",
                    "You can appear yourself — a lawyer is not required.",
                ],
            ),
        ]


register(ConsumerComplaintJourney())
