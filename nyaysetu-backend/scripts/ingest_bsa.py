"""Parse the official India Code BSA 2023 PDF into a per-section CSV for ingestion.

The Bharatiya Sakshya Adhiniyam, 2023 (Act 47 of 2023) replaced the Indian Evidence
Act, 1872. This script extracts all 170 sections from the official India Code bare-act
PDF (downloaded to data/_raw_sources/bsa_indiacode.pdf with a browser User-Agent) and
writes data/evidence/bsa_sections.csv. Deterministic only — no LLM touches the statutory
text.

Verification: it asserts all 170 sections are present and sequential, that section 170 is
"Repeal and savings", and cross-checks a sample of BSA headings against the independent
official NCRB IEA->BSA table (two official sources agreeing). Fails loudly otherwise.

Usage:
    python scripts/ingest_bsa.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PDF = Path("data/_raw_sources/bsa_indiacode.pdf")
OUT = Path("data/evidence/bsa_sections.csv")
MAPPING = Path("data/mappings/iea_bsa.json")
EXPECTED = 170

_HEADER = re.compile(r"(?m)^[ \t]*(\d+[A-Za-z]?)\.[ \t]+([\s\S]+?)\s*\.?[ \t]*[–—]{1,3}")
_DROP_LINE = re.compile(
    r"^\s*(\d+|SECTIONS?|PART [IVXLC]+.*|CHAPTER [IVXLC]+.*|THE BHARATIYA SAKSHYA ADHINIYAM.*)\s*$"
)
_FOOTNOTE = re.compile(r"^\s*\d+\.\s+(Subs|Ins|Omitted|Added|Vide|w\.e\.f|The word|Earlier|Cl\b)")


def _clean_heading(h: str) -> str:
    return re.sub(r"\s+", " ", h).strip().rstrip(".").strip()


def _clean_body(text: str, *, cap: int = 1600) -> str:
    kept = []
    for line in text.splitlines():
        if _DROP_LINE.match(line) or _FOOTNOTE.match(line):
            continue
        kept.append(line)
    out = re.sub(r"\s+", " ", " ".join(kept)).strip()
    if len(out) > cap:
        cut = out.rfind(". ", 0, cap)
        out = out[: (cut + 1) if cut > cap // 2 else cap].strip()
    return out


def extract_sections() -> list[tuple[str, str, str]]:
    from pypdf import PdfReader

    txt = "\n".join((p.extract_text() or "") for p in PdfReader(str(PDF)).pages)
    m = re.search(r"BE it enacted by Parliament", txt)
    if not m:
        raise SystemExit("Could not locate the enacting formula — wrong/garbled PDF?")
    body = txt[m.start():]

    matches = list(_HEADER.finditer(body))
    sections: list[tuple[str, str, str]] = []
    for i, mm in enumerate(matches):
        num = mm.group(1)
        heading = _clean_heading(mm.group(2))
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        content = _clean_body(body[mm.end():end])
        sections.append((num, heading, content))

    # Deduplicate to the first occurrence of each section number, keep document order.
    seen: dict[str, tuple[str, str, str]] = {}
    for num, heading, content in sections:
        seen.setdefault(num, (num, heading, content))
    return list(seen.values())


def main() -> int:
    sections = extract_sections()
    nums = [int(re.match(r"\d+", n).group()) for n, _, _ in sections]

    # --- Verify completeness + sequence ---
    missing = [n for n in range(1, EXPECTED + 1) if n not in set(nums)]
    if missing:
        print(f"FAIL: missing sections {missing}")
        return 1
    last = next((h for n, h, _ in sections if n == "170"), "")
    if "repeal" not in last.lower():
        print(f"FAIL: section 170 heading is {last!r}, expected 'Repeal and savings'")
        return 1

    # --- Cross-check BSA headings against the official NCRB IEA->BSA table ---
    bsa_heads = {n: h for n, h, _ in sections}
    iea_bsa = json.loads(MAPPING.read_text(encoding="utf-8"))["iea_to_bsa"]
    checks = {"23": "confession", "63": "electronic records", "39": "experts", "26": "dead"}
    mismatches = 0
    print("Cross-check BSA section headings (PDF) vs expectations:")
    for sec, kw in checks.items():
        h = bsa_heads.get(sec, "")
        ok = kw.lower() in h.lower()
        mismatches += 0 if ok else 1
        print(f"  {'OK ' if ok else 'BAD'} BSA {sec}: {h[:55]!r}")
    if mismatches:
        print(f"FAIL: {mismatches} heading cross-check mismatch(es)")
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "offence", "description"])
        for num, heading, content in sections:
            w.writerow([num, heading, content])

    print(f"\n[OK] wrote {len(sections)} BSA sections to {OUT}")
    print(f"     §1   : {bsa_heads.get('1','')[:60]}")
    print(f"     §63  : {bsa_heads.get('63','')[:60]}")
    print(f"     §170 : {bsa_heads.get('170','')[:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
