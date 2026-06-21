"""Parse the BNSS First Schedule (Part I — offences under the BNS) into a structured
classification table:  section, offence, punishment, cognizable, bailable, court.

WHY: offence classification (cognizable/bailable/triable-by) is the one thing the case
analysis must NOT guess — it isn't in the section text, only in this Schedule. Parsing it
from the official BNSS PDF turns those labels from "omitted" into "grounded".

HOW: the Schedule has no grid lines, so columns are reconstructed from word x-coordinates
(stable across the whole Schedule). Rows are anchored by a section number in column 1;
continuation lines append to the current row's columns.

TRUST: parsed from the official India Code BNSS PDF (data/_raw_sources/bnss.pdf). The
printed spot-checks MUST be eyeballed against the PDF before the output is trusted. Never
hand-author or "fix" a classification — re-parse from source.

    $env:PYTHONIOENCODING="utf-8"; python scripts/ingest_bnss_first_schedule.py
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

PDF = Path("data/_raw_sources/bnss.pdf")
OUT = Path("data/procedure/bnss_first_schedule.csv")

# Column x0 dividers discovered from the Schedule layout → 6 columns (idx 0..5).
DIVIDERS = (100, 220, 322, 385, 456)
SEC_ANCHOR = re.compile(r"^\d+[A-Za-z]?(?:\(\d+[A-Za-z]?\))?$")  # 89, 90(1), 303A


def col_of(x0: float) -> int:
    for i, d in enumerate(DIVIDERS):
        if x0 < d:
            return i
    return len(DIVIDERS)  # 5


def find_part1_range(pdf) -> tuple[int, int]:
    start = end = None
    for i, page in enumerate(pdf.pages):
        u = (page.extract_text() or "").upper()
        if start is None and "OFFENCES UNDER THE BHARATIYA NYAYA SANHITA" in u:
            start = i
        elif start is not None and "OFFENCES UNDER OTHER LAWS" in u:
            end = i
            break
    if start is None:
        raise SystemExit("Could not find Part I of the First Schedule in the PDF.")
    return start, (end if end is not None else len(pdf.pages))


def parse(pdf, start: int, end: int) -> list[dict]:
    rows: list[dict] = []
    cur: dict | None = None
    for pi in range(start, end):
        words = pdf.pages[pi].extract_words(use_text_flow=False, keep_blank_chars=False)
        lines: dict[int, list] = {}
        for w in words:
            lines.setdefault(round(w["top"] / 2) * 2, []).append(w)
        for key in sorted(lines):
            cols = ["", "", "", "", "", ""]
            for w in sorted(lines[key], key=lambda w: w["x0"]):
                c = col_of(w["x0"])
                cols[c] = (cols[c] + " " + w["text"]).strip()
            joined = " ".join(c for c in cols if c).strip()
            if not joined:
                continue
            U = joined.upper()
            # Skip repeated headers / titles / notes.
            if cols[:6] == ["1", "2", "3", "4", "5", "6"]:
                continue
            if U.startswith("SECTION OFFENCE") or "EXPLANATORY NOTES" in U:
                continue
            if "FIRST SCHEDULE" in U or "CLASSIFICATION OF OFFENCES" in U or "OFFENCES UNDER THE BHARATIYA" in U:
                continue
            c1 = cols[0].strip()
            # New row: a section anchor in col 1 WITH content elsewhere (a lone number is a
            # page number, not a section).
            if c1 and SEC_ANCHOR.match(c1) and any(cols[1:]):
                if cur:
                    rows.append(cur)
                cur = {
                    "section": c1, "offence": cols[1], "punishment": cols[2],
                    "cognizable": cols[3], "bailable": cols[4], "court": cols[5],
                }
            elif cur:
                for name, idx in (("offence", 1), ("punishment", 2), ("cognizable", 3),
                                  ("bailable", 4), ("court", 5)):
                    if cols[idx]:
                        cur[name] = (cur[name] + " " + cols[idx]).strip()
    if cur:
        rows.append(cur)
    return rows


_COG = {"cognizable": "Cognizable", "non-cognizable": "Non-cognizable"}
_BAIL = {"bailable": "Bailable", "non-bailable": "Non-bailable"}
_COURT = {
    "anymagistrate": "Any Magistrate",
    "magistrateofthefirstclass": "Magistrate of the first class",
    "magistrateofthesecondclass": "Magistrate of the second class",
    "courtofsession": "Court of Session",
}


def _norm(s: str, mapping: dict[str, str]) -> str | None:
    """Map a cell to ONE canonical value, tolerating the PDF's letter-spacing and
    repeated sub-rows. Returns the value only if every recognized sub-entry AGREES;
    returns None on a genuine conflict (e.g. theft: cognizable AND non-cognizable) or
    when nothing recognized — i.e. suppress unless unambiguous."""
    seen: set[str] = set()
    for seg in (s or "").split("."):
        key = re.sub(r"\s+", "", seg).lower()
        if key in mapping:
            seen.add(mapping[key])
    return next(iter(seen)) if len(seen) == 1 else None


def norm_cog(s: str) -> str | None:
    return _norm(s, _COG)


def norm_bail(s: str) -> str | None:
    return _norm(s, _BAIL)


def norm_court(s: str) -> str | None:
    return _norm(s, _COURT)


def main() -> int:
    with pdfplumber.open(str(PDF)) as pdf:
        start, end = find_part1_range(pdf)
        print(f"First Schedule Part I: pages {start}..{end - 1}")
        rows = parse(pdf, start, end)

    # One row per base section (sub-row classifications collapse onto it — which makes
    # multi-classification sections normalize to None below, i.e. safely suppressed).
    base = lambda r: re.match(r"^\d+", r["section"]).group()
    by: dict[str, dict] = {}
    for r in rows:
        m = re.match(r"^\d+", r["section"])
        if m and int(m.group()) <= 358:
            by.setdefault(m.group(), r)

    out_rows = []
    clean_count = 0
    for sec, r in sorted(by.items(), key=lambda kv: int(kv[0])):
        cog, bail, court = norm_cog(r["cognizable"]), norm_bail(r["bailable"]), norm_court(r["court"])
        # "Clean" = an unambiguous cognizable AND bailable value (court optional). Anything
        # conditional/multi/garbled normalizes to None and is marked ambiguous -> suppressed.
        unambiguous = bool(cog and bail)
        if unambiguous:
            clean_count += 1
        out_rows.append({
            "section": sec,
            "cognizable": cog or "",
            "bailable": bail or "",
            "court": court or "",
            "unambiguous": "yes" if unambiguous else "no",
            "raw_cognizable": " ".join(r["cognizable"].split())[:80],
            "raw_bailable": " ".join(r["bailable"].split())[:80],
            "raw_court": " ".join(r["court"].split())[:80],
            "offence": " ".join(r["offence"].split())[:120],
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["section", "cognizable", "bailable", "court", "unambiguous",
                                          "raw_cognizable", "raw_bailable", "raw_court", "offence"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {len(out_rows)} sections -> {OUT}")
    print(f"  unambiguous (safe to surface): {clean_count}   ambiguous/suppressed: {len(out_rows) - clean_count}\n")

    print("=== SPOT-CHECKS (verify against the official PDF) ===")
    for sec, name in [("64", "rape"), ("103", "murder"), ("115", "voluntarily causing hurt"),
                      ("303", "theft"), ("309", "robbery"), ("318", "cheating"), ("351", "criminal intimidation")]:
        r = next((x for x in out_rows if x["section"] == sec), None)
        if r:
            tag = "" if r["unambiguous"] == "yes" else "  [AMBIGUOUS -> suppressed]"
            print(f"BNS {sec} ({name}): {r['cognizable'] or r['raw_cognizable']} / "
                  f"{r['bailable'] or r['raw_bailable']} / {r['court'] or r['raw_court']}{tag}")
        else:
            print(f"BNS {sec} ({name}): NOT FOUND")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
