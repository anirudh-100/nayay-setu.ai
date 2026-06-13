"""Offline smoke test for RTI drafting (/draft/rti) — no downloads, no Ollama.

Substitutes a canned LLM (and a deliberately failing one) for OllamaClient, so it proves
the drafting orchestration is correct without any heavy externals:
  - the LLM's questions land in the assembled application letter,
  - the legal scaffolding (Section 6(1), fee, time limits, appeals) is present and correct,
  - the BPL fee exemption path is honoured,
  - and a model failure degrades gracefully to the citizen's raw request (still filable).

Usage:
    python scripts/rti_smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.llm_service import LLMError  # noqa: E402

_failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global _failures
    if not ok:
        _failures += 1
    line = f"  [{'PASS' if ok else 'FAIL'}] {name}"
    if detail:
        line += f"  - {detail}"
    print(line)


class FakeRTILLM:
    """Returns canned RTI-shaped JSON, mimicking OllamaClient.generate_json."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def warmup(self) -> None:  # pragma: no cover
        pass

    def generate_json(self, prompt: str) -> dict:
        return dict(self._payload)


class FailingLLM:
    """Simulates Ollama being unavailable."""

    def warmup(self) -> None:  # pragma: no cover
        pass

    def generate_json(self, prompt: str) -> dict:
        raise LLMError("simulated outage")


def main() -> int:
    print("RTI drafting smoke test (offline)\n" + "=" * 72)
    from app.knowledge import rti as kb
    from app.services.rti_service import RTIService

    canned = {
        "subject_line": "Status of passport application",
        "questions": [
            "Please provide the current status of passport application file no. XYZ.",
            "Please provide certified copies of the file notings on this application.",
            "Please state the expected date of dispatch of the passport.",
        ],
    }

    # --- Knowledge base sanity ---
    print("\nKnowledge base (app.knowledge.rti):")
    check("application fee is Rs.10 (central)", kb.APPLICATION_FEE_INR == 10)
    check("response limit is 30 days", kb.RESPONSE_DAYS == 30)
    check("life/liberty is 48 hours", kb.LIFE_LIBERTY_HOURS == 48)
    check("first appeal 30 / second 90 days", kb.FIRST_APPEAL_DAYS == 30 and kb.SECOND_APPEAL_DAYS == 90)
    cites = kb.core_citations()
    check("core citations cover s.6/7/19", {c.label for c in cites} ==
          {"RTI Act Section 6", "RTI Act Section 7", "RTI Act Section 19"},
          ", ".join(sorted(c.label for c in cites)))
    check("citations are curated + current + sourced",
          all(c.verification == "curated" and c.code_status == "current" and c.source_authority for c in cites))

    # --- Scenario A: full draft with authority + name (central, non-BPL) ---
    print("\nScenario A - full RTI draft (central, non-BPL):")
    a = RTIService(llm=FakeRTILLM(canned)).draft(
        subject="I want to know the status of my passport application",
        public_authority="Regional Passport Office, Pune",
        applicant_name="Asha Patil",
        applicant_address="12 MG Road, Pune 411001",
    )
    check("questions came from LLM", len(a.questions) == 3)
    check("subject line set", a.subject_line == "Status of passport application", a.subject_line)
    check("cites Section 6(1) in letter", "Section 6(1)" in a.application_text)
    check("letter addresses the PIO", "Public Information Officer" in a.application_text)
    check("first question in letter", canned["questions"][0] in a.application_text)
    check("applicant name in letter", "Asha Patil" in a.application_text)
    check("authority in letter", "Regional Passport Office, Pune" in a.application_text)
    check("non-BPL fee line (Rs.10)", "₹10" in a.application_text and "Below Poverty Line" not in a.application_text)
    check("filing has central portal", a.filing.portal == kb.RTI_ONLINE_PORTAL, str(a.filing.portal))
    check("timeline + appeals + tips present", a.timeline and a.appeals and a.tips)
    check("confidence high (llm ok + authority)", a.confidence == "high", a.confidence)
    check("citation_verified true", a.citation_verified is True)

    # --- Scenario B: BPL applicant -> fee exemption path ---
    print("\nScenario B - BPL applicant (fee exemption):")
    b = RTIService(llm=FakeRTILLM(canned)).draft(
        subject="Copies of muster rolls for MGNREGA work in my village",
        public_authority="Gram Panchayat Office",
        is_bpl=True,
    )
    check("letter cites BPL exemption s.7(5)", "Below Poverty Line" in b.application_text and "7(5)" in b.application_text)
    check("filing marks bpl exempt", b.filing.is_bpl_exempt is True)
    check("no Rs.10 fee demanded of BPL applicant", "₹10 is enclosed" not in b.application_text)

    # --- Scenario C: state level fee guidance ---
    print("\nScenario C - state-level request (fee varies):")
    c = RTIService(llm=FakeRTILLM(canned)).draft(
        subject="Status of my caste certificate application",
        level="state",
    )
    check("state fee guidance mentions State rules", "State" in c.filing.fee, c.filing.fee[:40])
    check("no central portal for state", c.filing.portal is None)
    check("confidence medium (no authority given)", c.confidence == "medium", c.confidence)

    # --- Scenario D: LLM outage -> graceful fallback ---
    print("\nScenario D - LLM unavailable (graceful fallback):")
    d = RTIService(llm=FailingLLM()).draft(
        subject="I want my property tax payment receipts for the last 3 years",
        public_authority="Municipal Corporation",
    )
    check("falls back to one question", len(d.questions) == 1)
    check("raw request used as question", "property tax" in d.questions[0])
    check("letter still valid (Section 6(1))", "Section 6(1)" in d.application_text)
    check("confidence medium (authority only)", d.confidence == "medium", d.confidence)

    # --- Scenario E: no authority, LLM outage -> low confidence + placeholder ---
    print("\nScenario E - no authority + LLM outage (low confidence, placeholder):")
    e = RTIService(llm=FailingLLM()).draft(subject="I want information about a road repair contract")
    check("authority placeholder used", "[Name of the public authority" in e.public_authority)
    check("name placeholder used", "[Your full name]" in e.application_text)
    check("confidence low", e.confidence == "low", e.confidence)

    print("=" * 72)
    if _failures == 0:
        print("ALL CHECKS PASSED")
    else:
        print(f"{_failures} CHECK(S) FAILED")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
