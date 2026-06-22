"""Offline unit tests for the reporter-citation scrubber (audit: fabrication-adjacent risk).

Runs with NO live LLM / index: we hand the deterministic scrubber canned
(answer text, retrieved-chunk texts) pairs and assert the offending precise law-report
citation string (SCC / SCR / AIR) is removed UNLESS it is present verbatim in a source.

Covers the four required cases:
  (a) a reporter string NOT in any source        -> removed
  (b) a reporter string present verbatim in a src -> kept
  (c) a plain case NAME with no reporter string    -> untouched
  (d) a normal answer with no citations            -> unchanged
...plus edge cases: parenthetical removal, AIR, bare SCC, dotted S.C.R., spacing
tolerance, multiple citations (one grounded one not), and sentence readability.

    $env:PYTHONIOENCODING="utf-8"; python scripts/citation_scrub_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.rag.models import Chunk, RetrievedChunk  # noqa: E402
from app.services.rag_service import _scrub_reporter_citations  # noqa: E402


def src(text):
    return RetrievedChunk(
        chunk=Chunk.create(text=text, source_type="judgment", ref="t", code_status="current"),
        score=1.0,
    )


passed = failed = 0


def check(name, cond, got=None):
    global passed, failed
    mark = "PASS" if cond else "FAIL"
    if cond:
        passed += 1
    else:
        failed += 1
    extra = "" if cond or got is None else f"   got: {got!r}"
    print(f"[{mark}] {name}{extra}")


NO_CITE_SOURCES = [src("Whoever cheats shall be punished with imprisonment up to seven years and fine.")]

# --- (a) reporter string NOT in sources -> removed ----------------------------------- #
# SCC
a1 = _scrub_reporter_citations(
    "The Court held that cheating requires dishonest inducement (2024) 10 SCC 1.",
    NO_CITE_SOURCES,
)
check("(a) SCC token not in sources is removed", "SCC" not in a1, a1)
check("(a) SCC scrub leaves sentence readable", "dishonest inducement" in a1 and a1.endswith("."), a1)

# SCR (square brackets, dotted)
a2 = _scrub_reporter_citations(
    "The principle is well settled [2024] 1 S.C.R. 1134 and applies here.",
    NO_CITE_SOURCES,
)
check("(a) SCR token not in sources is removed", "S.C.R" not in a2 and "1134" not in a2, a2)
check("(a) SCR scrub keeps surrounding clause", "well settled" in a2 and "applies here" in a2, a2)

# AIR
a3 = _scrub_reporter_citations(
    "This was decided in AIR 2024 SC 567 by the apex court.",
    NO_CITE_SOURCES,
)
check("(a) AIR token not in sources is removed", "AIR" not in a3 and "567" not in a3, a3)

# Bare SCC (no brackets)
a4 = _scrub_reporter_citations(
    "See 2024 SCC 1 for the rule.",
    NO_CITE_SOURCES,
)
check("(a) bare '2024 SCC 1' is removed", "SCC" not in a4, a4)

# --- (b) reporter string IS present verbatim in a source -> kept --------------------- #
grounded_src = [src("In Sharma v. State, reported at (2024) 10 SCC 1, the Court discussed cheating.")]
b1 = _scrub_reporter_citations(
    "The Court held that cheating requires dishonest inducement (2024) 10 SCC 1.",
    grounded_src,
)
check("(b) SCC token present verbatim in a source is KEPT", "(2024) 10 SCC 1" in b1, b1)

# Spacing tolerance: source has it, answer spaced slightly differently -> still kept.
grounded_air = [src("The leading authority is AIR 2024 SC 567 on this point.")]
b2 = _scrub_reporter_citations(
    "As established in AIR 2024 SC 567, the rule is clear.",
    grounded_air,
)
check("(b) AIR token present in a source is KEPT", "AIR 2024 SC 567" in b2, b2)

# --- (c) plain case NAME, no reporter string -> untouched ---------------------------- #
c1_in = "The Court in Sharma v. State of Bihar discussed dishonest inducement."
c1 = _scrub_reporter_citations(c1_in, NO_CITE_SOURCES)
check("(c) plain case name with no reporter string is untouched", c1 == c1_in, c1)

# --- (d) normal answer with no citations -> unchanged -------------------------------- #
d1_in = (
    "Under BNS Section 318, cheating is punishable with imprisonment up to seven years "
    "and a fine. File a police complaint with all your evidence."
)
d1 = _scrub_reporter_citations(d1_in, NO_CITE_SOURCES)
check("(d) normal answer with no citations is unchanged", d1 == d1_in, d1)

# --- edge: citation wrapped in its own parenthetical -> whole husk removed ----------- #
e1 = _scrub_reporter_citations(
    "Cheating requires dishonest inducement (see AIR 2024 SC 567).",
    NO_CITE_SOURCES,
)
check("(e) parenthetical-only citation husk removed, no empty ()", "(" not in e1 and "AIR" not in e1, e1)
check("(e) sentence still ends cleanly with period", e1.rstrip().endswith("."), e1)

# --- edge: two citations, one grounded + one not -> drop only the ungrounded one ----- #
mixed_src = [src("Authority: (2024) 10 SCC 1 is the governing precedent.")]
e2 = _scrub_reporter_citations(
    "The rule comes from (2024) 10 SCC 1 and was echoed in AIR 2024 SC 567.",
    mixed_src,
)
check("(f) grounded SCC kept", "(2024) 10 SCC 1" in e2, e2)
check("(f) ungrounded AIR dropped", "AIR" not in e2 and "567" not in e2, e2)

# --- edge: reasoning-style string is scrubbed too ------------------------------------ #
e3 = _scrub_reporter_citations(
    "Used BNS Section 318 from context; cf (2024) 5 SCC 99.",
    NO_CITE_SOURCES,
)
check("(g) reasoning string scrub keeps the grounding stem", "Used BNS Section 318 from context" in e3, e3)
check("(g) reasoning ungrounded SCC removed", "SCC" not in e3, e3)

# --- edge: empty / None input never raises ------------------------------------------- #
check("(h) empty string returns empty", _scrub_reporter_citations("", NO_CITE_SOURCES) == "")
check("(h) empty sources list is tolerated (removes ungrounded token)",
      "SCC" not in _scrub_reporter_citations("Rule from (2024) 10 SCC 1.", []))

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
