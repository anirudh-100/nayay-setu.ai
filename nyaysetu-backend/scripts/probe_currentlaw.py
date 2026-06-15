"""Probe: what does retrieval surface for the current-law failure cases, and would a
repealed-reference scan catch the signal needed to inject the current successor?

Prints, for each query, every retrieved chunk (type/act/section/snippet) and the
(code, section) repealed references a candidate scanner finds in the query and in
each chunk's text+title. Tells us whether query/guide scanning is enough to pull the
current statute into context — before wiring it into the engine.

Run with the API server STOPPED (embedded Qdrant is single-process).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.rag.law_map import LawMap  # noqa: E402
from app.rag.retriever import HybridRetriever  # noqa: E402

_NUM = r"(\d+[A-Za-z]?)"


def scan_repealed_refs(text: str, from_codes: list[str]) -> set[tuple[str, str]]:
    if not text:
        return set()
    found: set[tuple[str, str]] = set()
    for code in from_codes:
        c = re.escape(code)
        # CODE [section] N   — e.g. "IPC 420", "CrPC Section 154", "IPC s. 124A"
        for m in re.finditer(rf"\b{c}\b[\s.,]*(?:section|sec\.?|s\.?)?\s*{_NUM}", text, re.IGNORECASE):
            found.add((code, m.group(1)))
        # Section N [of [the]] CODE — e.g. "Section 154(3) CrPC", "section 420 of the IPC"
        for m in re.finditer(rf"(?:section|sec\.?|s\.?)\s*{_NUM}[^.\n]*?\bof\s+(?:the\s+)?\b{c}\b", text, re.IGNORECASE):
            found.add((code, m.group(1)))
        for m in re.finditer(rf"(?:section|sec\.?|s\.?)\s*{_NUM}\s*\(?\d*\)?\s*\b{c}\b", text, re.IGNORECASE):
            found.add((code, m.group(1)))
    return found


def main() -> int:
    law_map = LawMap.instance()
    from_codes = law_map.from_codes()
    print(f"from_codes: {from_codes}")
    r = HybridRetriever()

    for q in ["How do I file an FIR?", "What is the punishment for sedition under IPC 124A?"]:
        print("\n" + "=" * 78)
        print(f"QUERY: {q}")
        print(f"  query scan -> {scan_repealed_refs(q, from_codes)}")
        print("-" * 78)
        for rc in r.retrieve(q):
            c = rc.chunk
            blob = f"{c.text}\n{c.title or ''}"
            refs = scan_repealed_refs(blob, from_codes)
            print(f"  [{rc.score:6.2f}] type={c.source_type:8} act={str(c.act):5} sec={str(c.section):6} "
                  f"| {c.reference_label()[:40]}")
            if refs:
                succ = {f"{code} {sec}->{law_map.successor(code, sec)}" for code, sec in refs}
                print(f"           scan refs: {refs}")
                for s in succ:
                    print(f"           {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
