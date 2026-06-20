"""Capture full /ask answers for the gold questions, for model-vs-model comparison.

answer_eval.py gives a coarse pass/fail; this dumps the FULL answer + citation so two
engines (e.g. Haiku vs Opus) can be compared on quality the keyword eval can't see
(citation precision, spurious sections, nuance).

Usage:
    python scripts/capture_answers.py <out.json> [label]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import httpx  # noqa: E402

from answer_eval import API_URL, GOLD  # noqa: E402


def main() -> int:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "answers.json"
    label = sys.argv[2] if len(sys.argv) > 2 else "model"
    results = []
    with httpx.Client(timeout=180) as c:
        for case in GOLD:
            d = c.post(f"{API_URL}/ask", json={"query": case["q"], "language": "en"}).json()
            results.append({
                "id": case["id"],
                "q": case["q"],
                "law_reference": d.get("law_reference"),
                "answer": d.get("answer"),
                "current_law_note": d.get("current_law_note"),
                "citation_verified": d.get("citation_verified"),
                "confidence": d.get("confidence"),
                "response_time_ms": d.get("response_time_ms"),
            })
            print(f"  captured {case['id']}")
    Path(out_path).write_text(
        json.dumps({"label": label, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
