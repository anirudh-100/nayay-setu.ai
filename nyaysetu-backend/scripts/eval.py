"""Evaluate retrieval quality against the golden set — quality as a number, not a vibe.

For each golden query we check whether the *expected* section (or its IPC/BNS
counterpart) shows up in the retrieved chunks, and at what rank. Reports:

  - **Hit@k**  — fraction of queries whose expected section was retrieved in the top-k.
  - **MRR**    — mean reciprocal rank of the first correct chunk (rewards ranking it high).

This needs the built index (run ``python scripts/build_index.py`` first) but NOT Ollama —
it measures the retriever, which is where answer quality is won or lost. Use it to catch
regressions whenever you change embeddings, the reranker, fusion, or the corpus.

Usage:
    python scripts/eval.py                 # retrieval metrics
    python scripts/eval.py --top-k 8
    python scripts/eval.py --e2e           # also run the full LLM answer (needs Ollama)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.utils.logger import get_logger  # noqa: E402

logger = get_logger("eval")

_NUM_RE = re.compile(r"(\d+[A-Za-z]?)")


def _base(section: str) -> str:
    m = _NUM_RE.search(section or "")
    return m.group(1) if m else ""


def _load_golden(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=str(Path(settings.data_dir) / "eval" / "golden.jsonl"))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--e2e", action="store_true", help="also run the full RAG answer (needs Ollama)")
    args = parser.parse_args()

    from app.rag.retriever import HybridRetriever

    golden = _load_golden(Path(args.golden))
    retriever = HybridRetriever()
    rag = None
    if args.e2e:
        from app.services.rag_service import RAGService

        rag = RAGService()

    hits = 0
    rr_sum = 0.0
    print(f"\nEvaluating {len(golden)} queries (top_k={args.top_k})\n" + "-" * 72)

    for row in golden:
        query = row["query"]
        expect = {_base(s) for s in row["expect"]}
        results = retriever.retrieve(query, top_k=args.top_k)

        rank = None
        hit_label = ""
        for i, rc in enumerate(results, start=1):
            sec = _base(rc.chunk.section or "")
            if sec and sec in expect:
                rank = i
                hit_label = rc.chunk.reference_label()
                break

        if rank:
            hits += 1
            rr_sum += 1.0 / rank
            status = f"PASS  @{rank}  ({hit_label})"
        else:
            top = results[0].chunk.reference_label() if results else "—"
            status = f"FAIL        (top was: {top})"
        print(f"  [{status:<40}] {row['topic']:<22} | {query[:40]}")

        if rag is not None:
            ans = rag.answer(query)
            print(f"         ↳ answer law_ref={ans.law_reference!r} conf={ans.confidence} verified={ans.citation_verified}")

    n = len(golden) or 1
    print("-" * 72)
    print(f"  Hit@{args.top_k}: {hits}/{len(golden)} = {hits / n:.1%}")
    print(f"  MRR:     {rr_sum / n:.3f}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
