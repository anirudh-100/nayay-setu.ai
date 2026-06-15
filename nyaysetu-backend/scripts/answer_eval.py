"""Answer-quality eval: grade the FULL /ask pipeline, not just retrieval.

scripts/eval.py measures retrieval (did the right section rank in the top-k). This grades
the *answer the citizen actually gets* — the thing the product's trust promise rests on.

For each gold question it calls the live POST /ask endpoint and checks:
  - TRUST (hard fails — the failures that break a legal product):
      * citation_verified must be True on a substantive answer (no section cited that
        wasn't retrieved — the hallucination gate),
      * a question with no legal answer must ABSTAIN / escalate and must NOT fabricate a
        specific section.
  - QUALITY (soft checks): law_reference names the expected section (BNS or the IPC it
    replaced), the answer mentions an expected term, and the current-law bridge fired.

Run the API first (uvicorn app.main:app) — this hits it over HTTP, so no index lock fight
and it tests the real serving path. Exits non-zero if any TRUST hard-fail occurs.

Usage:
    python scripts/answer_eval.py
    API_URL=http://127.0.0.1:8000 python scripts/answer_eval.py
"""
from __future__ import annotations

import os
import sys

import httpx

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")

# Gold set. expect_section / expect_answer_any are lists of acceptable substrings (case-
# insensitive); the engine may cite the current BNS/BNSS/BSA section OR the repealed IPC/
# CrPC/IEA one (corpus has both), so we accept either. should_abstain marks non-legal
# questions that must NOT be answered with a fabricated section.
GOLD = [
    {"id": "murder", "q": "What is the punishment for murder?",
     "section": ["103", "302"], "answer_any": ["death", "imprisonment for life", "life"], "current": ["103", "302"]},
    {"id": "cheating", "q": "What is the punishment for cheating someone of money?",
     "section": ["318", "420"], "answer_any": ["cheat", "fraud", "deceiv", "property", "year"]},
    {"id": "fir", "q": "How do I file an FIR?",
     "section": ["173", "154"], "answer_any": ["police", "information", "cognizable", "fir", "officer"]},
    {"id": "anticipatory_bail", "q": "Can I get anticipatory bail?",
     "section": ["482", "438"], "answer_any": ["bail", "arrest", "court"]},
    {"id": "confession_police", "q": "Is a confession made to the police admissible as evidence?",
     "section": ["23", "25"], "answer_any": ["confession", "police", "not", "admissib"]},
    {"id": "dowry_death", "q": "What is the law on dowry death?",
     "section": ["80", "304B", "304"], "answer_any": ["dowry", "death", "year", "seven"]},
    {"id": "rape", "q": "What is the punishment for rape?",
     "section": ["64", "63", "376"], "answer_any": ["rape", "imprisonment", "year"]},
    {"id": "attempt_murder", "q": "What is the punishment for attempt to murder?",
     "section": ["109", "307"], "answer_any": ["attempt", "murder", "imprisonment"]},
    {"id": "theft", "q": "What is the punishment for theft?",
     "section": ["303", "378", "379"], "answer_any": ["theft", "imprisonment", "fine", "year"]},
    {"id": "grievous_hurt", "q": "Punishment for causing grievous hurt with a dangerous weapon?",
     "section": ["117", "118", "326"], "answer_any": ["hurt", "weapon", "imprisonment"]},
    {"id": "sedition", "q": "What is the punishment for sedition under IPC 124A?",
     "section": ["124A", "152"], "answer_any": ["sedition", "repeal", "no longer", "different", "152"]},
    {"id": "abstain_gold", "q": "What is the price of gold today?", "should_abstain": True},
    {"id": "abstain_sports", "q": "Who won the cricket match yesterday?", "should_abstain": True},
]


def _ask(q: str) -> dict:
    with httpx.Client(timeout=150) as c:
        r = c.post(f"{API_URL}/ask", json={"query": q, "language": "en"})
        r.raise_for_status()
        return r.json()


def _has_section(text: str, sections: list[str]) -> bool:
    t = text.lower()
    return any(s.lower() in t for s in sections)


def main() -> int:
    try:
        httpx.Client(timeout=8).get(f"{API_URL}/health").raise_for_status()
    except Exception as e:
        print(f"Backend not reachable at {API_URL} ({e}). Start it: uvicorn app.main:app")
        return 2

    print(f"Answer-quality eval over {len(GOLD)} questions @ {API_URL}\n" + "=" * 74)
    passes = 0
    trust_fails: list[str] = []
    quality_fails: list[str] = []

    for case in GOLD:
        try:
            resp = _ask(case["q"])
        except Exception as e:
            print(f"  [ERROR] {case['id']:<18} | request failed: {e}")
            trust_fails.append(f"{case['id']}: request failed")
            continue

        law_ref = str(resp.get("law_reference", ""))
        answer = str(resp.get("answer", ""))
        verified = bool(resp.get("citation_verified", True))
        abstained = bool(resp.get("abstained", False))
        conf = str(resp.get("confidence", ""))
        escalation = resp.get("escalation")
        cur_note = str(resp.get("current_law_note") or "")

        reasons: list[str] = []
        hard = False

        if case.get("should_abstain"):
            # Must not confidently answer a non-legal question with a fabricated section.
            abstained_ok = abstained or conf == "low" or bool(escalation)
            looks_fabricated = (not verified) or (
                conf == "high" and any(ch.isdigit() for ch in law_ref)
                and law_ref.lower() not in ("general legal guidance", "")
            )
            if looks_fabricated:
                hard = True
                reasons.append(f"fabricated/over-confident on non-legal Q (ref={law_ref!r}, conf={conf})")
            elif not abstained_ok:
                reasons.append(f"did not abstain/escalate (conf={conf})")
        else:
            if not verified:
                hard = True
                reasons.append(f"citation NOT verified — possible hallucinated section (ref={law_ref!r})")
            if not _has_section(law_ref + " " + answer, case["section"]):
                reasons.append(f"expected section {case['section']} not in answer (ref={law_ref!r})")
            if case.get("answer_any") and not any(a.lower() in answer.lower() for a in case["answer_any"]):
                reasons.append("answer missing all expected terms")
            if case.get("current") and not _has_section(cur_note, case["current"]):
                reasons.append(f"current-law bridge {case['current']} missing (note={cur_note[:40]!r})")

        ok = not reasons
        if ok:
            passes += 1
        elif hard:
            trust_fails.append(f"{case['id']}: " + "; ".join(reasons))
        else:
            quality_fails.append(f"{case['id']}: " + "; ".join(reasons))

        status = "PASS" if ok else ("TRUST-FAIL" if hard else "quality-miss")
        print(f"  [{status:<12}] {case['id']:<18} | ref={law_ref[:34]}")

    print("=" * 74)
    print(f"  PASS: {passes}/{len(GOLD)}")
    if trust_fails:
        print(f"\n  *** {len(trust_fails)} TRUST FAILURE(S) (hallucination / bad abstention) ***")
        for f in trust_fails:
            print(f"    - {f}")
    if quality_fails:
        print(f"\n  {len(quality_fails)} quality miss(es) (wrong/weak answer, not a trust break):")
        for f in quality_fails:
            print(f"    - {f}")
    print("\n" + ("RESULT: trust-clean" if not trust_fails else "RESULT: TRUST FAILURES PRESENT"))
    return 1 if trust_fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
