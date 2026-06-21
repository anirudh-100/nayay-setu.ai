"""Parse the official Consumer Protection Act, 2019 (India Code, Act 35 of 2019) into a
section-anchored CSV (section, title, description) the act loader can ingest.

WHY: the index had only a short plain-language consumer guide (no section anchors), so
pure-consumer queries retrieved weakly and ABSTAINED, and answers couldn't cite a CPA
section. This grounds them in the real Act.

HOW: the 2019 Act is cleanly typeset — every operative section is "N. Title.—body" (em
dash U+2014). We anchor on those headers (monotonically increasing section numbers, which
drops the arrangement-of-sections TOC and stray in-body matches) and take the text between
headers as the body.

TRUST: text is parsed from the official India Code PDF (data/_raw_sources/cpa_2019.pdf,
https://www.indiacode.nic.in/bitstream/123456789/15256/1/eng201935.pdf) — never hand-typed.
ONE faithful exception: the pecuniary-jurisdiction limits in ss.34/47/58 (enacted as ₹1cr /
₹1–10cr / >₹10cr) were superseded by the Consumer Protection (Jurisdiction of the District,
State and National Commission) Rules, 2021. Presenting the enacted figure alone would
misinform a citizen today, so those three sections carry an attributed note with the current
limits (₹50 lakh / ₹50 lakh–₹2cr / >₹2cr) — the same figures already in the repo's curated
consumer guide. Spot-checks MUST be eyeballed against the PDF before the output is trusted.

    $env:PYTHONIOENCODING="utf-8"; python scripts/ingest_cpa_2019.py
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

PDF = Path("data/_raw_sources/cpa_2019.pdf")
OUT = Path("data/acts/consumer_protection_act.csv")
MAX_BODY = 1800  # cap a section body so one chunk stays embeddable; trims trailing sub-clauses

# Current pecuniary jurisdiction (Consumer Protection (Jurisdiction ...) Rules, 2021), which
# superseded the enacted ss.34/47/58 limits. Figures match data/corpus/consumer_protection.md.
_JURISDICTION_NOTE = {
    "34": " [Current limit — as revised by the Consumer Protection (Jurisdiction of the "
          "District, State and National Commission) Rules, 2021: the District Commission "
          "hears complaints where the value of goods/services paid does not exceed ₹50 lakh.]",
    "47": " [Current limit — as revised by the 2021 Jurisdiction Rules: the State Commission "
          "hears complaints where the value exceeds ₹50 lakh but does not exceed ₹2 crore.]",
    "58": " [Current limit — as revised by the 2021 Jurisdiction Rules: the National "
          "Commission hears complaints where the value exceeds ₹2 crore.]",
}

# A section header is "N. Title.—". The title may WRAP across line breaks before the em
# dash, so the title class excludes only the dashes (negated classes also match newlines)
# and is length-bounded so it can never run into the next section. It stops at the first
# em/en dash, which inside a header is always the title's terminating ".—".
HEADER = re.compile(r"(?m)^(\d{1,3})\.[ \t]+([^–—]{1,300}?)\.\s*[–—]")


def main() -> int:
    with pdfplumber.open(str(PDF)) as pdf:
        full = "\n".join((p.extract_text() or "") for p in pdf.pages)

    m = re.search(r"BE it enacted by Parliament", full)
    body = full[m.end():] if m else full

    # Strip page furniture so it doesn't bleed into section bodies.
    kept = []
    for ln in body.split("\n"):
        s = ln.strip()
        if not s:
            continue
        if re.fullmatch(r"\d{1,3}", s):                       # page number
            continue
        if "THE CONSUMER PROTECTION ACT, 2019" in s.upper():  # running header
            continue
        kept.append(ln)
    body = "\n".join(kept)

    matches = list(HEADER.finditer(body))
    rows: list[dict] = []
    last = 0
    for i, mt in enumerate(matches):
        num = int(mt.group(1))
        if num <= last:           # monotonic: drops TOC echoes + stray in-body "N." lines
            continue
        last = num
        title = re.sub(r"\s+", " ", mt.group(2)).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        desc = re.sub(r"\s+", " ", body[mt.end():end]).strip()
        # The header line's own ".—" was consumed; re-stitch the title into the body so the
        # chunk reads naturally, then cap.
        desc = desc[:MAX_BODY].strip()
        note = _JURISDICTION_NOTE.get(str(num))
        if note:
            desc = (desc + note).strip()
        rows.append({"section": str(num), "title": title, "description": desc})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["section", "title", "description"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} sections -> {OUT} (range {rows[0]['section']}..{rows[-1]['section']})")

    print("\n=== SPOT-CHECKS (verify against the official PDF) ===")
    for want, label in [("2", "definitions"), ("34", "District jurisdiction"),
                        ("35", "how to complain"), ("38", "procedure"), ("39", "reliefs"),
                        ("47", "State jurisdiction"), ("58", "National jurisdiction"),
                        ("69", "limitation"), ("82", "product liability"), ("90", "penalty")]:
        r = next((x for x in rows if x["section"] == want), None)
        if r:
            print(f"\nCPA {want} ({label}) :: {r['title'][:70]}\n  {r['description'][:240]}")
        else:
            print(f"\nCPA {want} ({label}): NOT FOUND")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
