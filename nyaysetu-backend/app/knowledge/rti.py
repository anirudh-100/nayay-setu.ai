"""Source-attributed knowledge of the Right to Information Act, 2005.

This module is the single source of truth for the *procedural facts* the RTI drafter
relies on — the governing sections, the application fee, the statutory time limits, and
the appeal ladder. It is **hand-authored from the official RTI Act, 2005** and the RTI
(Regulation of Fee and Cost) Rules, 2005; **no LLM touches these facts**. The drafter's
language model only phrases the citizen's request into specific questions — every legal
number and citation below comes from here, deterministically.

Provenance & honesty (the product's core promise):
  - The section snippets are *plain-language paraphrases* compiled from the official Act,
    not verbatim bare-act text, so every RTI citation is marked ``verification="curated"``
    (surfaced as "Curated — confirm before relying"), with the official India Code source
    attached. Promote to ``official`` later by ingesting the bare RTI Act text.
  - Fee/timeline figures are the **Central Government** defaults under the 2005 Fee Rules.
    State governments notify their own RTI rules, so **state fees and portals vary** — the
    drafter says this explicitly rather than asserting a single number as universal.

Key sources:
  - RTI Act, 2005 (Act 22 of 2005): https://www.indiacode.nic.in/handle/123456789/2065
  - RTI (Regulation of Fee and Cost) Rules, 2005 (Central).
  - Central online filing portal: https://rtionline.gov.in
"""
from __future__ import annotations

from urllib.parse import quote_plus

from app.rag.models import Citation

# --- Canonical links ------------------------------------------------------- #
RTI_ACT_INDIA_CODE = "https://www.indiacode.nic.in/handle/123456789/2065"
RTI_ONLINE_PORTAL = "https://rtionline.gov.in"  # Central public authorities
NALSA_HELPLINE = "15100"

# --- Central-default figures (state rules vary) ---------------------------- #
APPLICATION_FEE_INR = 10          # s.6 + Fee Rules, 2005 (Central application fee)
COPY_FEE_PER_PAGE_INR = 2         # Fee Rules, 2005 (A4/A3 photocopy, per page)
RESPONSE_DAYS = 30                # s.7(1): ordinary time limit
LIFE_LIBERTY_HOURS = 48           # s.7(1) proviso: life or liberty of a person
APIO_EXTRA_DAYS = 5               # s.5(2): +5 days when filed via an APIO
THIRD_PARTY_DAYS = 40             # s.11: where third-party info is involved
FIRST_APPEAL_DAYS = 30            # s.19(1): file first appeal within 30 days
SECOND_APPEAL_DAYS = 90           # s.19(3): file second appeal within 90 days

_AUTHORITY = "Right to Information Act, 2005 (compiled from the official Act, India Code)"


def _ik(query: str) -> str:
    """Stable Indian Kanoon search link for an RTI Act section."""
    return f"https://indiankanoon.org/search/?formInput={quote_plus(query)}"


def _cite(section: str, snippet: str) -> Citation:
    """Build a curated, source-attributed RTI Act citation."""
    return Citation(
        label=f"RTI Act Section {section}",
        source_type="statute",
        snippet=snippet,
        url=_ik(f"section {section} right to information act 2005"),
        code_status="current",
        verification="curated",
        source_authority=_AUTHORITY,
    )


# Plain-language paraphrases of the provisions the drafter leans on. Compiled from the
# official Act — concise, accurate, NOT claimed as verbatim text.
_SECTIONS: dict[str, str] = {
    "6": (
        "Any citizen may request information in writing or electronically to the Public "
        "Information Officer (PIO) of the concerned public authority, giving the particulars "
        "of the information sought. No reason for seeking the information need be given, "
        "beyond contact details."
    ),
    "7": (
        "The PIO must provide the information or reject the request as soon as possible, and "
        "in any case within 30 days of receipt. Where the information concerns the life or "
        "liberty of a person, it must be provided within 48 hours. If the PIO misses the "
        "deadline, the information is provided free of charge."
    ),
    "8": (
        "Lists the categories of information that are exempt from disclosure (e.g. national "
        "security, matters before a court, cabinet papers, personal information with no public "
        "interest). A request may be refused only on these grounds."
    ),
    "18": (
        "Empowers the Information Commission to receive and inquire into complaints — for "
        "example, where no PIO was designated, a request was refused, or no response was given."
    ),
    "19": (
        "If you receive no reply within the time limit or are not satisfied, you may file a "
        "first appeal within 30 days to the officer senior to the PIO (the First Appellate "
        "Authority). A second appeal lies to the Central or State Information Commission "
        "within 90 days of the first-appeal decision."
    ),
}


def section_citation(section: str) -> Citation:
    """Citation for a single RTI Act section the drafter referenced."""
    return _cite(section, _SECTIONS.get(section, ""))


def core_citations() -> list[Citation]:
    """The sections every RTI draft rests on: the request, the time limit, and appeals."""
    return [section_citation(s) for s in ("6", "7", "19")]


def filing_guidance(*, level: str, is_bpl: bool) -> dict:
    """How and where to file, and what it costs. ``level`` is 'central' or 'state'."""
    central = level == "central"
    if is_bpl:
        fee_line = (
            "You are exempt from the application fee as a Below Poverty Line (BPL) applicant "
            "(RTI Act s.7(5)). Attach a copy of your BPL certificate / proof."
        )
    elif central:
        fee_line = (
            f"Application fee: ₹{APPLICATION_FEE_INR} (Central). Pay by Indian Postal "
            "Order (IPO), demand draft, or banker's cheque payable to the Accounts Officer of "
            "the public authority — or online if filing on the RTI portal. Copies cost about "
            f"₹{COPY_FEE_PER_PAGE_INR} per page."
        )
    else:
        fee_line = (
            "Application fee is set by your State's RTI Rules (commonly ₹10–₹50). "
            "Check your State Information Commission's website for the exact fee and accepted "
            "payment mode (often a court-fee stamp, IPO, or DD)."
        )

    if central:
        where = (
            "File with the Public Information Officer (PIO) of the public authority that holds "
            "the information. For most Central Government bodies you can file online at "
            f"{RTI_ONLINE_PORTAL}. Otherwise, send a signed application by registered post."
        )
    else:
        where = (
            "File with the Public Information Officer (PIO) of the State public authority that "
            "holds the information. Many States have their own RTI portal; otherwise send a "
            "signed application by registered post and keep the posting receipt."
        )

    return {
        "where_to_file": where,
        "fee": fee_line,
        "is_bpl_exempt": is_bpl,
        "portal": RTI_ONLINE_PORTAL if central else None,
    }


def timeline_guidance() -> list[str]:
    """The statutory clocks the applicant should expect, in plain language."""
    return [
        f"Normal reply: within {RESPONSE_DAYS} days of the PIO receiving your application "
        "(RTI Act s.7(1)).",
        f"Life or liberty matters: within {LIFE_LIBERTY_HOURS} hours (s.7(1) proviso).",
        f"If filed through an Assistant PIO, add {APIO_EXTRA_DAYS} days (s.5(2)).",
        "If the PIO misses the deadline, the information must be given free of charge (s.7(6)).",
    ]


def appeal_guidance() -> list[str]:
    """The two-step appeal ladder when a request is ignored or refused."""
    return [
        f"First appeal: within {FIRST_APPEAL_DAYS} days of the decision (or of the deadline "
        "passing with no reply), to the First Appellate Authority — the officer senior to the "
        "PIO in the same public authority (RTI Act s.19(1)). No fee for the first appeal.",
        f"Second appeal: within {SECOND_APPEAL_DAYS} days of the first-appeal decision, to the "
        "Central Information Commission (CIC) or your State Information Commission (s.19(3)).",
        f"You can also complain to the Information Commission under s.18, or seek free legal "
        f"aid (NALSA helpline {NALSA_HELPLINE}).",
    ]


def drafting_tips() -> list[str]:
    """Do's and don'ts that materially raise the odds of a useful reply."""
    return [
        "Ask for specific information, documents, or file notings — not opinions or "
        "hypothetical questions. PIOs are only required to give information that exists on "
        "record.",
        "Keep one application to one subject/public authority; split unrelated requests.",
        "Number your questions so each must be answered separately.",
        "Ask for certified copies of the relevant documents where useful.",
        "Keep proof of filing (the online registration number or the registered-post receipt) "
        "and note the date — your appeal deadlines run from it.",
    ]
