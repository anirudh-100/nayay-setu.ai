"""Offline unit tests for Hindi draft localization (DraftingService._localize_hindi).

No live LLM/index: a fake LLM returns canned JSON. Verifies the draft is translated only
when the model actually returns Hindi, that legal refs/amounts survive, section tone is
preserved, and any failure cleanly falls back to the English draft (never half-translated).

    $env:PYTHONIOENCODING="utf-8"; python scripts/drafting_hindi_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.knowledge.drafting.base import RenderResult  # noqa: E402
from app.schemas.draft import DraftSection  # noqa: E402
from app.services.drafting_service import DraftingService  # noqa: E402
from app.services.llm_service import LLMError  # noqa: E402


class FakeLLM:
    def __init__(self, resp):
        self.resp = resp

    def generate_json(self, prompt):
        if isinstance(self.resp, Exception):
            raise self.resp
        return self.resp


EN = RenderResult(
    document_text="To,\nThe Public Information Officer\n\nUnder Section 6(1) of the RTI Act, 2005, "
    "I request the following information. Fee of Rs. 10 enclosed.",
    subject_line="Request for information under the RTI Act, 2005",
    key_points=["Copies of the order dated 01/01/2024", "Status of file no. ABC-123"],
)
SECS = [DraftSection(label="How to file", items=["Pay Rs. 10 fee", "Submit to the PIO"], tone="info")]

passed = failed = 0


def check(name, cond):
    global passed, failed
    print(("[PASS] " if cond else "[FAIL] ") + name)
    passed += bool(cond); failed += (not cond)


def svc_with(resp):
    return DraftingService(llm=FakeLLM(resp))


# 1. Proper Hindi response -> translated, refs/amounts preserved, tone kept.
HI = {
    "document_text": "सेवा में,\nलोक सूचना अधिकारी\n\nRTI Act, 2005 की धारा 6(1) के तहत, मैं निम्नलिखित "
    "जानकारी का अनुरोध करता हूँ। Rs. 10 का शुल्क संलग्न है।",
    "subject_line": "RTI Act, 2005 के तहत सूचना का अनुरोध",
    "key_points": ["01/01/2024 के आदेश की प्रतियाँ", "फ़ाइल सं. ABC-123 की स्थिति"],
    "sections": [{"label": "कैसे दाखिल करें", "items": ["Rs. 10 शुल्क का भुगतान करें", "PIO को जमा करें"]}],
}
loc = svc_with(HI)._localize_hindi(EN, SECS)
check("Hindi response -> localized (not None)", loc is not None)
if loc:
    res, secs = loc
    import re
    dev = re.compile(r"[ऀ-ॿ]")
    check("document is in Hindi (Devanagari present)", bool(dev.search(res.document_text)))
    check("law reference preserved (RTI Act, 2005)", "RTI Act, 2005" in res.document_text)
    check("section number preserved (Section 6(1))", "धारा 6(1)" in res.document_text or "Section 6(1)" in res.document_text)
    check("amount preserved (Rs. 10)", "Rs. 10" in res.document_text)
    check("date/ref preserved in key_points", any("01/01/2024" in k for k in res.key_points))
    check("section tone preserved (info)", secs[0].tone == "info")
    check("section label translated", secs[0].label == "कैसे दाखिल करें")

# 2. Model returns English (no Devanagari) -> None (keep English, no half-translation).
EN_RESP = {"document_text": "To, The PIO ... under Section 6 ...", "subject_line": "x", "key_points": [], "sections": []}
check("English-only model response -> None", svc_with(EN_RESP)._localize_hindi(EN, SECS) is None)

# 3. LLM raises -> None (graceful fallback).
check("LLM error -> None", svc_with(LLMError("boom"))._localize_hindi(EN, SECS) is None)

# 4. Hindi doc but section list shape mismatch -> keep original English sections.
HI_BAD_SECS = dict(HI, sections=[])  # length 0 != 1
loc4 = svc_with(HI_BAD_SECS)._localize_hindi(EN, SECS)
check("section shape mismatch -> keep original English sections",
      loc4 is not None and loc4[1] == SECS)

# 5. English path is never localized (wiring helper).
from app.services.drafting_service import _wants_hindi  # noqa: E402
check("_wants_hindi('en') is False", _wants_hindi("en") is False)
check("_wants_hindi('hi') is True", _wants_hindi("hi") is True)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
