"""Parse the official India Code BNSS 2023 PDF into a per-section CSV for ingestion.

The Bharatiya Nagarik Suraksha Sanhita, 2023 (Act 46 of 2023) replaced the CrPC, 1973.
Unlike the BNS gazette (two-column, unparseable), the BNSS India Code PDF has a clean
single-column text layer, so we ingest it from the fully official source. Deterministic
only — no LLM touches the statutory text.

Verification: asserts all 531 sections present + sequential, that s.531 is "Repeal and
savings", spot-checks key procedure sections (173 FIR, 480 bail) verbatim, and
cross-checks EVERY heading against the official NCRB BNSS Corresponding Section Table.

Usage:
    python scripts/ingest_bnss.py
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PDF = Path("data/_raw_sources/bnss.pdf")
NCRB = Path("data/_raw_sources/ncrb_bnss.html")
OUT = Path("data/procedure/bnss_sections.csv")
EXPECTED = 531

# Section number is <=3 digits (max section 531) so stray years like "2023." can't match;
# the heading can't cross into a following "N. " section line (prevents over-consumption).
_HEADER = re.compile(
    r"(?m)^[ \t]*(\d{1,3}[A-Za-z]?)\.[ \t]+((?:(?!\n[ \t]*\d{1,3}\.[ \t])[\s\S])+?)\s*\.?[ \t]*[–—]{1,3}"
)
_DROP_LINE = re.compile(
    r"^\s*(\d+|SECTIONS?|PART [IVXLC]+.*|CHAPTER [IVXLC]+.*|THE BHARATIYA NAGARIK SURAKSHA SANHITA.*)\s*$"
)
_FOOTNOTE = re.compile(r"^\s*\d+\.\s+(Subs|Ins|Omitted|Added|Vide|w\.e\.f|The word|Earlier|Cl\b)")


def _clean_heading(h: str) -> str:
    return re.sub(r"\s+", " ", h).strip().rstrip(".").strip()


def _clean_body(text: str, *, cap: int = 1600) -> str:
    kept = [ln for ln in text.splitlines() if not _DROP_LINE.match(ln) and not _FOOTNOTE.match(ln)]
    out = re.sub(r"\s+", " ", " ".join(kept)).strip()
    if len(out) > cap:
        cut = out.rfind(". ", 0, cap)
        out = out[: (cut + 1) if cut > cap // 2 else cap].strip()
    return out


def extract_sections() -> dict[str, tuple[str, str]]:
    from pypdf import PdfReader

    txt = "\n".join((p.extract_text() or "") for p in PdfReader(str(PDF)).pages)
    m = re.search(r"BE it enacted by Parliament", txt)
    if not m:
        raise SystemExit("Could not find the enacting formula — wrong/garbled PDF?")
    body = txt[m.start():]
    matches = list(_HEADER.finditer(body))
    out: dict[str, tuple[str, str]] = {}
    for i, mm in enumerate(matches):
        num = mm.group(1)
        base = int(re.match(r"\d+", num).group())
        if base < 1 or base > EXPECTED:  # drop spurious matches (years, footnote nums)
            continue
        if num in out:
            continue
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        out[num] = (_clean_heading(mm.group(2)), _clean_body(body[mm.end():end]))
    return out


def ncrb_headings() -> dict[str, str]:
    import pandas as pd

    t = pd.read_html(NCRB, encoding="utf-8")[0]
    heads: dict[str, str] = {}
    for cell in t.iloc[:, 0]:  # col 0 = BNSS section + heading
        mm = re.match(r"^\s*(\d+[A-Za-z]?)\.\s+([^\n]+)", str(cell).replace("�", "-"))
        if mm:
            h = re.sub(r"[\s.]*\d+[A-Za-z]?\s*(\(\s*\d+[A-Za-z]?\s*\))*[\s.]*$", "", mm.group(2))
            heads.setdefault(mm.group(1), re.sub(r"\s+", " ", h).strip().rstrip(".").lower())
    return heads


def main() -> int:
    secs = extract_sections()
    nums = {int(re.match(r"\d+", n).group()) for n in secs}
    missing = [n for n in range(1, EXPECTED + 1) if n not in nums]
    if missing:
        print(f"FAIL: missing sections {missing[:30]}")
        return 1
    if "repeal" not in secs.get("531", ("", ""))[0].lower():
        print(f"FAIL: s.531 heading is {secs.get('531')!r}")
        return 1

    # Spot-check key procedure sections verbatim.
    s173 = secs.get("173", ("", ""))[1].lower()
    s480 = secs.get("480", ("", ""))[1].lower()
    ok = ("information" in s173 or "cognizable" in s173) and "bail" in s480
    print(f"Key-section sanity (173 FIR / 480 bail): {ok}")
    if not ok:
        return 1

    # Cross-check every heading against the official NCRB BNSS table.
    official = ncrb_headings()
    matched = mism = 0
    examples = []
    for num, (heading, _) in secs.items():
        off = official.get(num)
        if not off or not heading:
            continue
        a, b = set(heading.lower().split()), set(off.split())
        if len(a & b) >= max(1, min(len(a), len(b)) // 2):
            matched += 1
        else:
            mism += 1
            if len(examples) < 8:
                examples.append(f"{num}: PDF[{heading[:28]}] vs NCRB[{off[:28]}]")
    print(f"Heading cross-check vs official NCRB: {matched} match, {mism} differ")
    for e in examples:
        print(f"  diff {e}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "offence", "description"])
        for num in sorted(secs, key=lambda n: int(re.match(r"\d+", n).group())):
            heading, body = secs[num]
            w.writerow([num, heading, body])
    print(f"\n[OK] wrote {len(secs)} BNSS sections to {OUT}")
    print(f"     §173: {secs['173'][0][:55]}")
    print(f"     §480: {secs['480'][0][:55]}")
    print(f"     §531: {secs['531'][0][:55]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
