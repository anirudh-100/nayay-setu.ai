"""Offline tests for the offence-classification fixes (audit majors #2/#3).

No live LLM/index: exercises OffenceClassification (suppression of the 318 sub-section
conflict) and RAGService._offence_classification (lead-only selection + death-deeming
deny-list), which together must STOP the three confirmed misclassifications:
  - cheating-with-delivery (318) labelled non-cognizable/bailable  -> now suppressed
  - living dowry victim labelled BNS 80 "Dowry Death"/Court of Session -> suppressed
  - morphed-photo lead 77 (voyeurism) falling through to lesser 351  -> suppressed
...while genuine single-offence classifications (103 murder, 305, 309) still surface.

    $env:PYTHONIOENCODING="utf-8"; python scripts/classification_fix_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.rag.offence_classification import OffenceClassification  # noqa: E402
from app.services.rag_service import RAGService  # noqa: E402

oc = OffenceClassification.instance()
clf = RAGService._offence_classification

passed = failed = 0


def check(name, cond):
    global passed, failed
    print(("[PASS] " if cond else "[FAIL] ") + name)
    passed += bool(cond); failed += (not cond)


# --- Classifier-level suppression / grounding ---
check("318 suppressed (sub-section conflict)", oc.classify("318") is None)
check("318(4) suppressed (base 318)", oc.classify("318(4)") is None)
check("85 stays None (garbled cognizable cell)", oc.classify("85") is None)
check("77 stays None (1st/2nd-conviction conflict)", oc.classify("77") is None)
check("103 murder still grounds", oc.classify("103") is not None)
check("305 theft-in-dwelling still grounds", oc.classify("305") is not None)
check("309 robbery still grounds", oc.classify("309") is not None)
check("117 grievous hurt still grounds (no collateral damage)", oc.classify("117") is not None)
check("80 dowry-death still grounds at classifier level", oc.classify("80") is not None)

# --- Selection-level (lead-only + deny-80) ---
check("cheating 318 -> no banner", clf("BNS Section 318", False) == "")
check("cheating 318(4) -> no banner", clf("BNS Section 318(4)", False) == "")
check("dowry, 80 listed first -> suppressed (deny death-deeming)", clf("BNS Section 80; BNS Section 85", False) == "")
check("dowry, 85 listed first -> suppressed (85 ambiguous, no fall-through to 80)",
      clf("BNS Section 85; BNS Section 80", False) == "")
check("morphed photo, lead 77 -> suppressed (no fall-through to 351)",
      clf("BNS Section 77; BNS Section 351", False) == "")
check("murder 103 -> banner shown", clf("BNS Section 103", False) != "")
check("murder 103 banner names the section", "Section 103" in clf("BNS Section 103", False))
check("theft-in-dwelling 305 -> banner shown", clf("BNS Section 305", False) != "")
check("robbery 309 -> banner shown", clf("BNS Section 309", False) != "")
check("grievous hurt 117 -> banner shown", clf("BNS Section 117", False) != "")
check("non-BNS headline (IPC) -> no BNS banner", clf("IPC Section 420", False) == "")
check("generic headline -> no banner", clf("General Legal Guidance", False) == "")
check("Hindi murder 103 -> Hindi banner (Devanagari)",
      any("ऀ" <= ch <= "ॿ" for ch in clf("BNS Section 103", True)))

# A pure criminal-intimidation case (351 as lead) is a legit standalone classification.
check("criminal intimidation 351 as lead -> banner shown", clf("BNS Section 351", False) != "")

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
