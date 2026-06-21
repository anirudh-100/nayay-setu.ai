"""Adversarial unit tests for RAGService._build_analysis (the case-analysis gate/scrub).

Runs offline (no model, no index): we construct fake RetrievedChunks and call the
deterministic builder directly. Each test encodes one finding from the design's
adversarial trust review — the block must stay inside the trust contract.

    $env:PYTHONIOENCODING="utf-8"; python scripts/analysis_build_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")  # Devanagari on Windows consoles
except Exception:
    pass

from app.rag.models import Chunk, RetrievedChunk  # noqa: E402
from app.services.rag_service import RAGService  # noqa: E402


def chunk(act, section, text="x"):
    return RetrievedChunk(chunk=Chunk.create(text=text, source_type="statute", ref=f"{act}-{section}",
                                             act=act, section=section, code_status="current"), score=1.0)


# A RAGService without touching the heavy singletons (pass truthy stubs).
svc = RAGService(llm=object(), retriever=object())

BNS318 = chunk("BNS", "318", "Whoever cheats shall be punished ... up to seven years and fine.")
BNSS173 = chunk("BNSS", "173", "Information in cognizable cases ... FIR.")
# Realistic: an offence query puts only the OFFENCE (BNS) in the model-visible context; the
# BNSS process arc arrives separately via _procedure_context (the `proc` argument).
VISIBLE = [BNS318]
PROC = [BNSS173]

passed = failed = 0


def check(name, cond):
    global passed, failed
    mark = "PASS" if cond else "FAIL"
    if cond:
        passed += 1
    else:
        failed += 1
    print(f"[{mark}] {name}")


def build(raw, *, conf="high", verified=True, ref="BNS Section 318", visible=VISIBLE, proc=PROC, hindi=False):
    return svc._build_analysis(raw, visible, list(proc), hindi, conf, verified, ref)


FULL = {
    "situation": ["This appears to involve cheating — taking money by deceit."],
    "applicable_law": [
        "BNS Section 318 — cheating — up to 7 years and fine (was IPC 420)",
        "BNS Section 999 — fabricated section not in context",
    ],
    "what_happens_next": [
        "File an FIR (BNSS Section 173).",
        "The matter then goes to trial and judgment.",  # no BNSS section -> dropped
    ],
    "do_now": ["Save every message, receipt, and transfer proof.", "Note all dates."],
    "also_possible": [
        "The law allows some such disputes to be compounded.",
        "You will definitely get bail.",                       # outcome prediction -> dropped
        "The law allows this cognizable offence to be settled.",  # classification -> dropped
    ],
    "for_your_advocate": [
        "Ingredients to prove: deception and dishonest inducement.",
        "See Sharma v. State of Bihar on cheating.",            # precedent -> dropped
    ],
}

# 1. Happy path: a strong, verified Mode-A answer yields a populated block.
a = build(FULL)
check("happy path returns CaseAnalysis", a is not None)

# 2. applicable_law: fabricated section (999) dropped; real one (318) kept.
check("applicable_law drops fabricated section 999",
      a and a.applicable_law == ["BNS Section 318 — cheating — up to 7 years and fine (was IPC 420)"])

# 3. what_happens_next: only steps citing a BNSS section survive — and the grounding
#    comes from the INJECTED procedure arc (proc=[BNSS173]), not the offence context.
check("what_happens_next keeps only the BNSS-grounded step (via injected procedure)",
      a and a.what_happens_next == ["File an FIR (BNSS Section 173)."])

# 3b. With NO procedure injected, a BNSS step has nothing to ground on -> dropped,
#     but the rest of the analysis still builds.
no_proc = build(FULL, proc=[])
check("what_happens_next empty when no procedure arc is injected",
      no_proc is not None and no_proc.what_happens_next == [])

# 4. also_possible: outcome prediction AND classification bullets dropped; safe stem kept.
check("also_possible keeps only the impersonal, non-classification bullet",
      a and a.also_possible == ["The law allows some such disputes to be compounded."])

# 5. for_your_advocate: precedent citation dropped; grounded pointer kept.
check("for_your_advocate strips the 'X v. Y' precedent",
      a and a.for_your_advocate == ["Ingredients to prove: deception and dishonest inducement."])

# 6. outcome_framing is set in code (never blank) on a non-null block.
check("outcome_framing populated in code", a and a.outcome_framing.startswith("This explains"))

# 7. Low confidence -> None (suppression gate).
check("low confidence suppresses analysis", build(FULL, conf="medium") is None)

# 8. Unverified citation -> None.
check("unverified citation suppresses analysis", build(FULL, verified=False) is None)

# 9. Generic / Mode-B headline -> None (even at high confidence).
check("generic ref suppresses analysis", build(FULL, ref="General Legal Guidance") is None)
check("Article ref suppresses analysis", build(FULL, ref="Article 21") is None)

# 10. Headline section not in visible context -> None.
check("headline section absent from context suppresses analysis",
      build(FULL, ref="BNS Section 555") is None)

# 11. Everything scrubbed away -> None (graceful fallback to the plain card).
poison_only = {"also_possible": ["You will win.", "This is a bailable offence."],
               "for_your_advocate": ["See Kumar v. State."]}
check("all-poison input collapses to None", build(poison_only) is None)

# 12. Hindi classification term (संज्ञेय) is stripped, not silently exempt.
hi_raw = {"situation": ["यह संज्ञेय अपराध प्रतीत होता है।"],          # classification -> dropped
          "do_now": ["हर संदेश और रसीद सुरक्षित रखें।"]}
ha = build(hi_raw, hindi=True)
check("Hindi classification bullet stripped", ha is not None and ha.situation == [])
check("Hindi safe bullet kept", ha is not None and ha.do_now == ["हर संदेश और रसीद सुरक्षित रखें।"])
check("Hindi outcome_framing localized", ha is not None and "भविष्यवाणी" in ha.outcome_framing)

# 13. A step citing a NON-BNSS section number present in context is still dropped
#     (procedure must be grounded in BNSS specifically). 318 is a BNS section here.
check("non-BNSS section does not ground a process step",
      build({"what_happens_next": ["Step under Section 318."], "do_now": ["Keep records."]})
      .what_happens_next == [])

# 14. Offence classification (BNSS First Schedule) — grounded, set in code, not the LLM.
cls103 = svc._offence_classification("BNS Section 103", False)  # murder
check("classification: murder (103) = cognizable/non-bailable/Sessions",
      "cognizable" in cls103.lower() and "non-bailable" in cls103.lower() and "session" in cls103.lower())
check("classification: theft (303) is conditional -> empty (suppressed)",
      svc._offence_classification("BNS Section 303", False) == "")
check("classification: non-BNS ref (Article 21) -> empty",
      svc._offence_classification("Article 21", False) == "")
check("classification: differing offences (103 + 318) -> empty (don't over-simplify)",
      svc._offence_classification("BNS Section 103 and BNS Section 318", False) == "")
check("classification: Hindi murder contains संज्ञेय",
      "संज्ञेय" in svc._offence_classification("BNS Section 103", True))
# 15. Classification flows into the built analysis for a single clear offence (318 cheating).
check("analysis.classification set for single clear offence (318)", bool(a and a.classification))

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
