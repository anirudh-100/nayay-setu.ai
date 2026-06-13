"""Offline smoke test for the generic drafting engine (/draft) — no Ollama, no downloads.

Exercises the registry and all three launch journeys (RTI, consumer complaint, police
complaint) with a canned LLM and a failing one, proving:
  - the framed content lands in the assembled document,
  - the right statutory sections / forum scaffolding are present and correct,
  - required-field validation and unknown-journey handling work,
  - and a model outage degrades gracefully to the citizen's raw input (still usable).

Usage:
    python scripts/drafting_smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Windows consoles default to cp1252 and choke on ₹ etc. in output; force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

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


class FakeLLM:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def warmup(self) -> None:  # pragma: no cover
        pass

    def generate_json(self, prompt: str) -> dict:
        return dict(self._payload)


class FailingLLM:
    def warmup(self) -> None:  # pragma: no cover
        pass

    def generate_json(self, prompt: str) -> dict:
        raise LLMError("simulated outage")


def main() -> int:
    print("Drafting engine smoke test (offline)\n" + "=" * 72)
    from app.services.drafting_service import (
        DraftingService,
        JourneyNotFound,
        MissingFields,
        list_journeys,
    )

    # --- Registry ---
    print("\nRegistry (GET /draft/journeys):")
    journeys = list_journeys()
    ids = [j.id for j in journeys]
    check("3 journeys registered", ids == ["rti", "consumer_complaint", "police_complaint"], ", ".join(ids))
    check("each journey exposes fields", all(len(j.fields) >= 1 for j in journeys))
    check("each journey has a doc_title + icon", all(j.doc_title and j.icon for j in journeys))

    # --- RTI ---
    print("\nRTI journey:")
    rti_llm = FakeLLM({"subject_line": "Status of passport application", "questions": [
        "Please provide the current status of my passport application dated __________.",
        "Please provide certified copies of all file notings on this application.",
    ]})
    r = DraftingService(llm=rti_llm).draft(
        journey_id="rti",
        fields={"subject": "status of my passport", "public_authority": "RPO Pune", "level": "central"},
        applicant_name="Asha Patil",
    )
    check("RTI cites Section 6(1)", "Section 6(1)" in r.document_text)
    check("RTI questions in key_points", len(r.key_points) == 2)
    check("RTI citations s.6/7/19", {c.label for c in r.citations} ==
          {"RTI Act Section 6", "RTI Act Section 7", "RTI Act Section 19"})
    check("RTI sections present", [s.label for s in r.sections] and any("file" in s.label.lower() for s in r.sections))
    check("RTI confidence high (llm + authority)", r.confidence == "high", r.confidence)
    rb = DraftingService(llm=rti_llm).draft(journey_id="rti",
                                            fields={"subject": "muster rolls", "is_bpl": True}, applicant_name="X")
    check("RTI BPL cites s.7(5)", "7(5)" in rb.document_text and "Below Poverty Line" in rb.document_text)

    # --- Consumer complaint ---
    print("\nConsumer complaint journey:")
    cc_llm = FakeLLM({
        "grievance_points": [
            "The Complainant purchased a phone from the Opposite Party on __________ for ₹15,000.",
            "The phone stopped working within a week and the Opposite Party refused to refund or replace it.",
        ],
        "reliefs": ["Direct the Opposite Party to refund ₹15,000.", "Pay compensation for harassment."],
    })
    c = DraftingService(llm=cc_llm).draft(
        journey_id="consumer_complaint",
        fields={"opposite_party": "ABC Electronics", "grievance": "phone broke, no refund", "amount": "15000"},
        applicant_name="Ravi Kumar",
    )
    check("CC cites Section 35 (filing)", "SECTION 35" in c.document_text.upper())
    check("CC cites Section 2(7) consumer", "Section 2(7)" in c.document_text)
    check("CC cites Section 69 limitation", "Section 69" in c.document_text)
    check("CC has prayer with reliefs", "PRAYER" in c.document_text and "refund ₹15,000" in c.document_text)
    check("CC citations include s.34 + s.69", {"Consumer Protection Act Section 34", "Consumer Protection Act Section 69"} <=
          {c2.label for c2 in c.citations})
    check("CC forum section mentions 50 lakh limit", any("50 lakh" in i for s in c.sections for i in s.items))
    check("CC confidence high", c.confidence == "high", c.confidence)

    # --- Police complaint ---
    print("\nPolice complaint journey:")
    pc_llm = FakeLLM({"facts": [
        "On __________ near __________, two men snatched my mobile phone and fled.",
        "The incident was witnessed by __________.",
    ]})
    p = DraftingService(llm=pc_llm).draft(
        journey_id="police_complaint",
        fields={"incident": "phone snatched", "station": "Shivajinagar PS"},
        applicant_name="Sara Khan",
    )
    check("PC requests FIR under Section 173", "Section 173 of the Bharatiya Nagarik Suraksha Sanhita" in p.document_text)
    check("PC facts in document", "snatched my mobile phone" in p.document_text)
    check("PC addressed to SHO", "Station House Officer" in p.document_text)
    check("PC citations include 173 + 175(3)", {"BNSS Section 173", "BNSS Section 175(3)"} <=
          {c3.label for c3 in p.citations})
    check("PC has refusal-escalation section", any("refuse" in s.label.lower() for s in p.sections))

    # --- Validation + errors ---
    print("\nValidation & errors:")
    try:
        DraftingService(llm=cc_llm).draft(journey_id="consumer_complaint", fields={"amount": "100"})
        check("missing required field rejected", False)
    except MissingFields as e:
        check("missing required field rejected", True, "; ".join(e.missing))
    try:
        DraftingService(llm=cc_llm).draft(journey_id="does_not_exist", fields={})
        check("unknown journey rejected", False)
    except JourneyNotFound:
        check("unknown journey rejected", True)

    # --- LLM outage fallback ---
    print("\nLLM outage (graceful fallback):")
    cf = DraftingService(llm=FailingLLM()).draft(
        journey_id="consumer_complaint",
        fields={"opposite_party": "XYZ", "grievance": "paid for a sofa never delivered"},
    )
    check("CC fallback uses raw grievance", "sofa never delivered" in cf.document_text)
    check("CC fallback still cites Section 35", "SECTION 35" in cf.document_text.upper())
    check("CC fallback confidence medium", cf.confidence == "medium", cf.confidence)

    # --- numeric-array defence (small-model quirk) ---
    print("\nNumeric-array defence:")
    n = DraftingService(llm=FakeLLM({"questions": [1, 2.2, 3]})).draft(
        journey_id="rti", fields={"subject": "my pension status", "public_authority": "EPFO"})
    check("numeric questions rejected -> fallback to subject", n.key_points == ["my pension status"], str(n.key_points))

    print("=" * 72)
    if _failures == 0:
        print("ALL CHECKS PASSED")
    else:
        print(f"{_failures} CHECK(S) FAILED")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
