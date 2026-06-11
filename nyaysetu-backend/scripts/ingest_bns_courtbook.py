"""Scrape the full BNS 2023 bare act (358 sections) from Courtbook and verify it.

The official India Code / MHA BNS PDF is a two-column gazette whose marginal headings
float detached from the section bodies, so naive PDF text extraction scrambles it.
Courtbook.in publishes the same act as clean, pre-segmented per-section HTML (the
provision text was spot-checked verbatim-correct against the official PDF). This script
harvests all 358 per-section links from the index (the opaque IDs can't be constructed),
fetches each page, and extracts the heading + provision text deterministically — no LLM
touches the statutory text.

TRUST: Courtbook is a secondary (non-government) source, so this script cross-checks
EVERY section heading against the official NCRB Corresponding Section Table (an MHA
source) and reports the match rate. The text remains pending line-by-line verification
against India Code; provenance records this honestly.

Usage:
    python scripts/ingest_bns_courtbook.py
"""
from __future__ import annotations

import csv
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bs4 import BeautifulSoup  # noqa: E402

INDEX = Path("data/_raw_sources/courtbook_bns_index.html")
NCRB_BNS = Path("data/_raw_sources/ncrb_bns.html")
OUT = Path("data/bns/bns_sections.csv")
BASE = "https://courtbook.in"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120 Safari/537.36"}
EXPECTED = 358
_SEC_IN_SLUG = re.compile(r"/section-(\d+[A-Za-z]?)-")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).replace("..", ".").strip()


def harvest_links() -> list[tuple[str, str]]:
    soup = BeautifulSoup(INDEX.read_text(encoding="utf-8", errors="replace"), "lxml")
    out: dict[str, str] = {}
    for a in soup.find_all("a", href=True):
        m = _SEC_IN_SLUG.search(a["href"])
        if m:
            out.setdefault(m.group(1), a["href"])
    return sorted(out.items(), key=lambda kv: int(re.match(r"\d+", kv[0]).group()))


def fetch_section(num: str, href: str) -> tuple[str, str, str]:
    url = BASE + href
    for attempt in range(3):
        try:
            html = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read()
            break
        except Exception:
            if attempt == 2:
                return num, "", ""
            time.sleep(0.6 * (attempt + 1))
    soup = BeautifulSoup(html, "lxml")
    h1 = soup.find("h1")
    heading = _norm(h1.get_text(" ", strip=True)).rstrip(".").strip() if h1 else ""
    # The provision text follows the "View Act" marker. Among all blocks containing it,
    # pick the one whose post-marker tail is longest (the real content, not a nav stub).
    best = ""
    for d in soup.find_all("div"):
        text = d.get_text(" ", strip=True)
        if "View Act" not in text:
            continue
        tail = re.split(r"View Act\s*[→>]*", text)[-1]
        if len(tail) > len(best):
            best = tail
    return num, heading, _norm(best)


def ncrb_headings() -> dict[str, str]:
    """Official BNS section -> heading from the NCRB table, for cross-checking."""
    import pandas as pd

    t = pd.read_html(NCRB_BNS, encoding="utf-8")[0]
    heads: dict[str, str] = {}
    for cell in t.iloc[:, 0]:
        m = re.match(r"^\s*(\d+[A-Za-z]?)\.\s+([^\n]+)", str(cell))
        if m:
            # Strip trailing subsection tokens / numbers that float into NCRB headings
            # (e.g. "Kidnapping. 137(1)" -> "kidnapping").
            h = re.sub(r"[\s.]*\d+[A-Za-z]?\s*(\(\s*\d+[A-Za-z]?\s*\))*[\s.]*$", "", m.group(2))
            heads.setdefault(m.group(1), _norm(h).rstrip(".").lower())
    return heads


def main() -> int:
    links = harvest_links()
    print(f"Harvested {len(links)} BNS section links from the index.")
    if len(links) < EXPECTED:
        print(f"WARNING: expected {EXPECTED} links, got {len(links)}")

    results: dict[str, tuple[str, str, str]] = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for num, heading, body in ex.map(lambda kv: fetch_section(*kv), links):
            results[num] = (num, heading, body)
    fetched = [r for r in results.values() if r[1] or r[2]]
    print(f"Fetched {len(fetched)}/{len(links)} section pages with content.")

    # --- Completeness ---
    nums = {int(re.match(r"\d+", n).group()) for n in results if results[n][1] or results[n][2]}
    missing = [n for n in range(1, EXPECTED + 1) if n not in nums]
    if missing:
        print(f"WARNING: missing/empty sections: {missing[:30]}{'...' if len(missing) > 30 else ''}")

    # --- Cross-check headings against the official NCRB table ---
    official = ncrb_headings()
    matched = mism = 0
    examples = []
    for num, (n, heading, body) in results.items():
        off = official.get(num)
        if not off or not heading:
            continue
        # Token-overlap match (Courtbook title-cases / abbreviates differently than NCRB).
        a, b = set(heading.lower().split()), set(off.split())
        if len(a & b) >= max(1, min(len(a), len(b)) // 2):
            matched += 1
        else:
            mism += 1
            if len(examples) < 8:
                examples.append(f"{num}: CB[{heading[:30]}] vs NCRB[{off[:30]}]")
    print(f"\nHeading cross-check vs official NCRB: {matched} match, {mism} differ")
    for e in examples:
        print(f"  diff {e}")

    # --- Sanity: a known punishment must be present verbatim ---
    s103 = results.get("103", ("", "", ""))[2]
    ok103 = "death or" in s103.lower() and "imprisonment for life" in s103.lower()
    print(f"\n§103 murder punishment present verbatim: {ok103}")
    if not ok103:
        print(f"  §103 body: {s103[:160]!r}")

    if missing or not ok103:
        print("\nFAIL: completeness/sanity check failed — not writing CSV.")
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "offence", "description"])
        for num, _ in links:
            n, heading, body = results[num]
            w.writerow([num, heading, body])
    print(f"\n[OK] wrote {len(links)} BNS sections to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
