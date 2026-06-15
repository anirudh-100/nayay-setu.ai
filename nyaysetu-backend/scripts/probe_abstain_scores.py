"""Measure the reranker's top-1 score for legal vs. off-topic queries.

The /ask path abstains when the top reranked score falls below MIN_RERANK_SCORE.
To set that threshold without over-abstaining on real legal questions, we need the
actual score distribution. This prints the top-1 rerank score for a spread of genuine
legal queries (must stay ABOVE the threshold) and off-topic ones (should fall BELOW),
so the gap between the two clusters tells us where the line belongs.

Run with the API server STOPPED (embedded Qdrant is single-process / file-locked):
    python scripts/probe_abstain_scores.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console: emit UTF-8
except Exception:
    pass

from app.rag.retriever import HybridRetriever  # noqa: E402

LEGAL = [
    "What is the punishment for murder?",
    "How do I file an FIR?",
    "Is a confession made to the police admissible as evidence?",
    "Can I get anticipatory bail?",
    "What is the law on dowry death?",
    "What is the punishment for cheating someone of money?",
    "Punishment for causing grievous hurt with a dangerous weapon?",
    "What are my rights if I am arrested?",
]
OFF_TOPIC = [
    "What is the price of gold today?",
    "Who won the cricket match yesterday?",
    "What is the weather forecast for tomorrow?",
    "Give me a good recipe for pizza.",
    "Which smartphone should I buy this year?",
    "How do I lose weight fast?",
]


def main() -> int:
    r = HybridRetriever()

    def probe(label: str, queries: list[str]) -> list[float]:
        print(f"\n{label}")
        print("-" * 70)
        tops: list[float] = []
        for q in queries:
            res = r.retrieve(q)
            if res:
                top = res[0]
                tops.append(top.score)
                lbl = top.chunk.reference_label()
                print(f"  {top.score:8.3f}  {q[:46]:<46}  -> {lbl}")
            else:
                print(f"  {'(none)':>8}  {q[:46]:<46}  -> no results")
        return tops

    legal = probe("LEGAL queries (must stay ABOVE threshold)", LEGAL)
    junk = probe("OFF-TOPIC queries (should fall BELOW threshold)", OFF_TOPIC)

    print("\n" + "=" * 70)
    if legal:
        print(f"  Legal     min={min(legal):8.3f}  max={max(legal):8.3f}")
    if junk:
        print(f"  Off-topic min={min(junk):8.3f}  max={max(junk):8.3f}")
    if legal and junk:
        gap_lo, gap_hi = max(junk), min(legal)
        if gap_lo < gap_hi:
            suggested = (gap_lo + gap_hi) / 2
            print(f"  CLEAN GAP: off-topic top {gap_lo:.3f} < legal bottom {gap_hi:.3f}")
            print(f"  Suggested MIN_RERANK_SCORE ~= {suggested:.2f}")
        else:
            print(f"  OVERLAP: off-topic max {gap_lo:.3f} >= legal min {gap_hi:.3f} "
                  f"— no clean threshold; need a secondary signal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
