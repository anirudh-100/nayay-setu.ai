"""Grave-offence current-law regression harness — the durable guard for offence mapping.

This is the protective net for the fragile, high-stakes mapping cases. The small model is
NON-DETERMINISTIC: on an attempt-to-murder fact pattern (victim ALIVE + intent to kill =
IPC 307 -> BNS 109) it sometimes mislabels the headline as completed murder (BNS 103),
theft (BNS 307 — "Theft after preparation…", which is the successor of IPC *382*, NOT
IPC 307), or under-charges to mere hurt. The single clearest, provable defect is the
"BNS 307 (was IPC 307)" footgun: BNS 307 is a THEFT offence per data/bns/bns_sections.csv,
so an attempt-on-life framing on it is internally inconsistent with the curated LawMap
(IPC 307's successor is BNS 109).

This harness has two layers:

  1. DETERMINISTIC (always runs, must be "N passed, 0 failed"): exercises the
     `_current_law_guard_violation` guard and `RAGService._offence_classification` directly
     with the headline strings each grave-offence archetype SHOULD and SHOULD NOT produce.
     It pins the guard's behaviour: it MUST fire on the contradictory headline (and suppress
     the grounded classification), and MUST NOT fire on any consistent mapping (attempt-to-
     murder 109, murder 103, grievous hurt 117/118, theft 303/305, robbery 309, cheating
     318) so no correct case is ever mislabelled. This is the regression-proof layer.

  2. LIVE (opt-in: `--live N`, makes real LLM + retrieval calls): runs the natural-language
     fact patterns through `RAGService.answer()` N times each and reports, per archetype,
     how often the headline landed on an acceptable BNS section / a safe (escalated)
     outcome. Because the model is non-deterministic this layer DOCUMENTS a flake rate
     rather than asserting a hard pass — the guard's job is to make the *worst* outcome
     (a confidently-asserted, internally-contradictory grave-offence label) impossible, not
     to make the model deterministic.

    $env:PYTHONIOENCODING="utf-8"; python scripts/grave_offence_regression_test.py
    $env:PYTHONIOENCODING="utf-8"; python scripts/grave_offence_regression_test.py --live 5
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.services.rag_service import (  # noqa: E402
    RAGService,
    _current_law_guard_violation,
)

clf = RAGService._offence_classification

passed = failed = 0


def check(name, cond):
    global passed, failed
    print(("[PASS] " if cond else "[FAIL] ") + name)
    passed += bool(cond)
    failed += (not cond)


# --------------------------------------------------------------------------- #
# LAYER 1 — DETERMINISTIC GUARD / CLASSIFICATION REGRESSION (must be all-green)
# --------------------------------------------------------------------------- #
# The contradictory headline the guard exists to catch. BNS 307 is a *theft* offence; the
# successor of IPC 307 (attempt to murder) is BNS 109. A "BNS 307 (was IPC 307)" headline
# is therefore internally inconsistent with the curated LawMap.
print("== guard fires on the internally-inconsistent grave-offence headline ==")
check("attempt-on-life mislabel: 'BNS 307 (was IPC 307)' flagged inconsistent",
      _current_law_guard_violation("BNS Section 307 (was IPC 307)") is not None)
check("attempt-on-life mislabel: its grounded classification is SUPPRESSED",
      clf("BNS Section 307 (was IPC 307)", False) == "")
check("any wrong-successor headline flagged: 'BNS 103 (was IPC 307)' inconsistent",
      _current_law_guard_violation("BNS Section 103 (was IPC 307)") is not None)
check("Hindi form still flagged: 'BNS धारा 307 (was IPC 307)'",
      _current_law_guard_violation("BNS धारा 307 (was IPC 307)") is not None)

# The guard must NOT fire — and classification must behave normally — on every CONSISTENT
# grave-offence mapping. This is what stops the guard from mislabelling correct cases.
print("\n== guard stays silent on every CONSISTENT mapping (no false positives) ==")
consistent_no_violation = [
    ("attempt to murder (correct)", "BNS Section 109 (was IPC 307)"),
    ("attempt to murder, no old ref", "BNS Section 109"),
    ("murder", "BNS Section 103 (was IPC 302)"),
    ("theft after preparation (true 307 successor of IPC 382)", "BNS Section 307 (was IPC 382)"),
    ("grievous hurt", "BNS Section 117 (was IPC 325)"),
    ("hurt by dangerous weapon", "BNS Section 118 (was IPC 324)"),
    ("theft in dwelling", "BNS Section 305 (was IPC 380)"),
    ("robbery", "BNS Section 309 (was IPC 392)"),
    ("cheating", "BNS Section 318(4) (was IPC 420)"),
    ("plain BNS headline, no parenthetical", "BNS Section 307"),
    ("non-BNS / repealed headline", "IPC Section 420"),
    ("generic", "General Legal Guidance"),
]
for label, ref in consistent_no_violation:
    check(f"no false positive: {label} ({ref!r})",
          _current_law_guard_violation(ref) is None)

# Grounded classification still surfaces for the genuinely-clear grave offences (the guard
# is surgical — it suppresses ONLY the contradictory headline, not these).
print("\n== correct grave-offence classifications still surface (guard didn't over-reach) ==")
check("attempt to murder 109 -> Sessions banner shown, names 109",
      clf("BNS Section 109 (was IPC 307)", False) != "" and "109" in clf("BNS Section 109 (was IPC 307)", False))
check("murder 103 -> banner shown, names 103",
      clf("BNS Section 103 (was IPC 302)", False) != "" and "103" in clf("BNS Section 103", False))
check("grievous hurt 117 -> banner shown", clf("BNS Section 117 (was IPC 325)", False) != "")
check("hurt by dangerous weapon 118 -> banner shown", clf("BNS Section 118 (was IPC 324)", False) != "")
check("theft-in-dwelling 305 -> banner shown", clf("BNS Section 305 (was IPC 380)", False) != "")
check("robbery 309 -> banner shown", clf("BNS Section 309 (was IPC 392)", False) != "")
# Theft 307 itself (true successor of IPC 382) is conditional/ambiguous at the schedule
# level, so the lead-only rule leaves it UNLABELLED — a safe outcome (never a wrong banner).
check("theft 307 (correct mapping) -> conditional, safely unlabelled",
      clf("BNS Section 307 (was IPC 382)", False) == "")
# Cheating 318 is suppressed for the known sub-section conflict (existing behaviour preserved).
check("cheating 318 -> suppressed (sub-section conflict, unchanged)",
      clf("BNS Section 318(4) (was IPC 420)", False) == "")


# --------------------------------------------------------------------------- #
# LAYER 2 — LIVE FACT-PATTERN COVERAGE (opt-in; documents the non-deterministic flake rate)
# --------------------------------------------------------------------------- #
# Each archetype: a natural-language situation, the BNS section(s) that are an ACCEPTABLE
# headline for it, and whether an escalated/abstained answer counts as a safe non-mislabel.
# "Acceptable" is generous on purpose — we are measuring how often the engine AVOIDS the
# grave mislabel, not pinning it to one exact section.
LIVE_CASES = [
    {
        "name": "attempt to murder (victim alive, intent to kill)",
        "query": ("My brother was attacked with a knife by a man who clearly wanted to kill "
                  "him. He survived but is in the ICU. What is the offence?"),
        # 109 = attempt to murder (ideal). 117/118 = grievous hurt / hurt by dangerous weapon
        # (an under-charge, but NOT the grave mislabel). The FORBIDDEN outcomes are 103
        # (completed murder — the victim is alive) and 307 (theft — wrong offence entirely).
        "acceptable": {"109", "117", "118"},
        "forbidden": {"103", "307"},
    },
    {
        "name": "murder (victim dead)",
        "query": "A man stabbed my neighbour to death during a fight. What law applies?",
        "acceptable": {"103", "100", "101"},   # 103 murder; 100/101 culpable homicide family
        "forbidden": {"307", "109"},            # not theft; victim is dead, not 'attempt'
    },
    {
        "name": "grievous hurt",
        "query": ("A man hit my father with an iron rod and broke his arm in a quarrel. "
                  "There was no intent to kill. What offence is this?"),
        "acceptable": {"115", "117", "118", "122"},
        "forbidden": {"307"},                   # not theft
    },
    {
        "name": "theft",
        "query": "Someone stole my mobile phone from my bag in a crowded market. What is the offence?",
        "acceptable": {"303", "304", "305"},    # theft family
        "forbidden": {"103", "109"},            # not a homicide offence
    },
    {
        "name": "robbery",
        "query": ("Two men stopped me on the road at night, showed a knife and forced me to "
                  "hand over my wallet and phone. What offence is this?"),
        "acceptable": {"309", "310"},           # robbery / dacoity
        "forbidden": {"103"},
    },
    {
        "name": "cheating",
        "query": ("A man took 5 lakh rupees from me promising to double it in a scheme, then "
                  "vanished and blocked my number. What offence is this?"),
        "acceptable": {"316", "318"},           # criminal breach of trust / cheating
        "forbidden": {"103", "109", "307"},
    },
]


def _bns_bases(law_reference):
    """Base BNS section numbers named in a headline string. Mirrors the extraction in
    RAGService._offence_classification ('BNS Section 109' -> {'109'})."""
    import re
    return {re.match(r"(\d+)", t).group(1)
            for t in re.findall(
                r"\bBNS\b\s*(?:Section|Sec\.?|S\.?|धारा)?\s*(\d+[A-Za-z]?)",
                law_reference or "", re.IGNORECASE)}


def run_live(repeats):
    print(f"\n== LIVE fact-pattern coverage ({repeats} run(s) each) — documents flake rate ==")
    svc = RAGService()
    grave_mislabels = 0
    for case in LIVE_CASES:
        ok = safe = mislabel = 0
        seen = []
        for _ in range(repeats):
            try:
                r = svc.answer(case["query"])
            except Exception as e:
                print(f"  [{case['name']}] run errored: {e}")
                continue
            bases = _bns_bases(r.law_reference)
            seen.append((r.law_reference, r.confidence, r.escalation is not None))
            forbidden_hit = bool(bases & case["forbidden"])
            # An escalated / low-confidence answer is a SAFE outcome (we did not vouch).
            escalated = r.escalation is not None or r.confidence == "low" or r.abstained
            if bases & case["acceptable"]:
                ok += 1
            elif escalated and not forbidden_hit:
                safe += 1
            elif forbidden_hit and not escalated:
                mislabel += 1
                grave_mislabels += 1
            else:
                safe += 1  # some other section, not forbidden, not asserted as grave
        print(f"  [{case['name']}] acceptable={ok} safe(other/escalated)={safe} GRAVE_MISLABEL={mislabel}")
        for ref, conf, esc in seen:
            print(f"       -> {ref!r} (conf={conf}, escalated={esc})")
    print(f"\n  LIVE summary: {grave_mislabels} grave mislabel(s) across "
          f"{repeats * len(LIVE_CASES)} run(s).")
    print("  NOTE: the model is non-deterministic; this layer documents the flake rate. The "
          "deterministic guard above is what makes a CONFIDENTLY-ASSERTED, internally-"
          "contradictory grave-offence label impossible.")
    return grave_mislabels


def main():
    live = 0
    if "--live" in sys.argv:
        i = sys.argv.index("--live")
        live = int(sys.argv[i + 1]) if i + 1 < len(sys.argv) else 3

    print(f"\n{passed} passed, {failed} failed  (deterministic layer)")
    if live:
        run_live(live)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
