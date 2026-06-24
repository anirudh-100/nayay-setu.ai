"""Generic: parse a cleanly-typeset Indian bare-act PDF (India Code) into a section-anchored
CSV (section, title, description) the act loader ingests. Generalised from
ingest_cpa_2019.py (same proven parser: "N. Title.—body" headers, em-dash delimiter,
monotonic section dedup that drops the arrangement-of-sections TOC + stray in-body matches).

    python scripts/ingest_act_pdf.py <pdf_path> <out_csv_path>

TRUST: text is parsed verbatim from the official India Code PDF — never hand-typed. ALWAYS
eyeball the printed spot-checks against the PDF before trusting the output.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pdfplumber  # noqa: E402

MAX_BODY = 1800
# A section header is "N. Title.—". Title may wrap across line breaks; the negated class
# also matches newlines and is length-bounded so it can't run into the next section. It
# stops at the first em/en dash, which in a header is always the title's terminating ".—".
HEADER = re.compile(r"(?m)^(\d{1,3})\.[ \t]+([^–—]{1,300}?)\.\s*[–—]")


def parse(pdf_path: str) -> list[dict]:
    with pdfplumber.open(pdf_path) as pdf:
        full = "\n".join((p.extract_text() or "") for p in pdf.pages)
    m = re.search(r"BE it enacted by Parliament", full)
    body = full[m.end():] if m else full

    kept = []
    for ln in body.split("\n"):
        s = ln.strip()
        if not s or re.fullmatch(r"\d{1,3}", s):
            continue
        kept.append(ln)
    body = "\n".join(kept)

    matches = list(HEADER.finditer(body))
    rows: list[dict] = []
    last = 0
    for i, mt in enumerate(matches):
        num = int(mt.group(1))
        if num <= last:          # monotonic: drops TOC echoes + stray in-body "N." lines
            continue
        last = num
        title = re.sub(r"\s+", " ", mt.group(2)).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        desc = re.sub(r"\s+", " ", body[mt.end():end]).strip()[:MAX_BODY]
        rows.append({"section": str(num), "title": title, "description": desc})
    return rows


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: ingest_act_pdf.py <pdf_path> <out_csv_path>")
        return 2
    pdf_path, out_csv = argv[1], argv[2]
    rows = parse(pdf_path)
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["section", "title", "description"])
        w.writeheader()
        w.writerows(rows)
    nums = [int(r["section"]) for r in rows]
    print(f"Wrote {len(rows)} sections -> {out} (range {min(nums)}..{max(nums)})")
    missing = [n for n in range(min(nums), max(nums) + 1) if n not in set(nums)]
    print(f"  missing in range: {missing if missing else 'none'}")
    print("\n=== SPOT-CHECKS (verify against the PDF) ===")
    for r in rows[:2] + rows[len(rows) // 2: len(rows) // 2 + 1] + rows[-2:]:
        print(f"  s.{r['section']} {r['title'][:60]} :: {r['description'][:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
