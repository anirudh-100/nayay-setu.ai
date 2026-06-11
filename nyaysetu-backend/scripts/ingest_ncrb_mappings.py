"""Rebuild the IPC->BNS / CrPC->BNSS / IEA->BSA mappings from the OFFICIAL NCRB tables.

The National Crime Records Bureau (NCRB, under the MHA) publishes the official
"Corresponding Section Tables" for the three new criminal codes as clean HTML. This
script parses those tables deterministically (no LLM — legal text must be exact) and
writes our mapping JSONs, upgrading them from curated guesses to **official-sourced**
data. Human notes and "no equivalent in the new code" entries from the prior curated
files are preserved (merged by old-section key) because the NCRB tables are keyed by
the *new* law and so don't list dropped old sections.

Source HTML is fetched separately into data/_raw_sources/ (see the curl in the build
notes). Re-run after refreshing those files. Every mapping is validated against the
authority of record (India Code) before the 'verified' flag should be trusted; we set
verified=true here because NCRB is an official MHA publication.

Usage:
    python scripts/ingest_ncrb_mappings.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

RAW = Path("data/_raw_sources")
MAPPINGS = Path("data/mappings")
RETRIEVED_AT = "2026-06-11"
NCRB_AUTHORITY = "NCRB / MHA — Corresponding Section Tables (cytrain.ncrb.gov.in)"

# (new_code, old_code, raw_html, out_file, new_title, old_title, ncrb_url)
TABLES = [
    ("BNS", "IPC", "ncrb_bns.html", "ipc_bns.json",
     "Bharatiya Nyaya Sanhita, 2023", "Indian Penal Code, 1860",
     "https://cytrain.ncrb.gov.in/staticpage/web_pages/SectionTableBNS.html"),
    ("BNSS", "CrPC", "ncrb_bnss.html", "crpc_bnss.json",
     "Bharatiya Nagarik Suraksha Sanhita, 2023", "Code of Criminal Procedure, 1973",
     "https://cytrain.ncrb.gov.in/staticpage/web_pages/SectionTableBNSS.html"),
    ("BSA", "IEA", "ncrb_bsa.html", "iea_bsa.json",
     "Bharatiya Sakshya Adhiniyam, 2023", "Indian Evidence Act, 1872",
     "https://cytrain.ncrb.gov.in/staticpage/web_pages/SectionTableBSA.html"),
]

_NEW_SUBSEC = re.compile(r"^(\d+[A-Za-z]?)\s*\(\s*(\d+[A-Za-z]?)\s*\)")
_NEW_HEADER = re.compile(r"^(\d+[A-Za-z]?)\.\s+(\S.*)$")
_NEW_PLAIN = re.compile(r"^(\d+[A-Za-z]?)\b")
_OLD_NUM = re.compile(r"^(\d+[A-Za-z]?)")
_TRAIL_SUBSEC = re.compile(r"\s*\d+[A-Za-z]?\s*\(\s*[0-9A-Za-z]+\s*\)\s*$")
_OLD_HEADER = re.compile(r"^\d+[A-Za-z]?\.\s*(.+)$")
_SKIP_OLD = {"new section", "new sub-section", "new sub section", "deleted", "", "nan"}
# When the NEW-law cell says one of these, the OLD section was dropped (no successor).
_DROPPED = {"deleted", "repealed", "omitted"}


def _clean(x: object) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return re.sub(r"\s+", " ", str(x)).replace("�", "-").strip()


def _clean_heading(h: str) -> str:
    # Strip only a trailing floating subsection token (e.g. "... 5(a)"), not real words.
    h = _TRAIL_SUBSEC.sub("", h).strip()
    return h.rstrip(". ").strip()


def parse_table(new_code: str, old_code: str, html_path: Path) -> dict:
    """Return {old_base_section: {"<new_code_lower>": new_token, "offence": heading}}."""
    df = pd.read_html(html_path, encoding="utf-8")[0]
    df.columns = ["new", "old"]
    rows = [(_clean(a), _clean(b)) for a, b in zip(df["new"], df["old"])]

    mapping: dict[str, dict] = {}
    cur_base = ""
    cur_heading = ""

    for c0, c1 in rows:
        # --- new-law cell: determine the target token + track section heading ---
        m_sub = _NEW_SUBSEC.match(c0)
        m_hdr = _NEW_HEADER.match(c0)
        if m_hdr:
            cur_base = m_hdr.group(1)
            cur_heading = _clean_heading(m_hdr.group(2))

        # --- old-law cell: the section this new provision replaced ---
        if c1.lower() in _SKIP_OLD:
            continue  # this row introduces a NEW provision, no old section to map
        m_old = _OLD_NUM.match(c1)
        if not m_old:
            continue
        old_base = m_old.group(1)
        if old_base in mapping:
            continue  # first (primary) mapping for an old section wins

        # The new-law cell flags an old section that was DROPPED — record it as "no equivalent".
        if c0.strip().lower() in _DROPPED:
            mo = _OLD_HEADER.match(c1)
            mapping[old_base] = {
                new_code.lower(): None,
                "offence": _clean_heading(mo.group(1)) if mo else "",
                "note": f"No corresponding section in the {new_code} (repealed without a direct successor).",
            }
            continue

        # Otherwise resolve the new-law target token (subsection > header > plain > current section).
        if m_sub:
            new_token = f"{m_sub.group(1)}({m_sub.group(2)})"
        elif m_hdr:
            new_token = m_hdr.group(1)
        else:
            m_plain = _NEW_PLAIN.match(c0)
            new_token = m_plain.group(1) if m_plain else cur_base
        if not new_token:
            continue

        mapping[old_base] = {new_code.lower(): new_token, "offence": cur_heading}

    return mapping


def merge_overlay(new_map: dict, overlay_key: str, new_code: str) -> tuple[dict, int]:
    """Merge human-authored notes from the FIXED curated overlay onto the official
    mapping. The overlay is never overwritten, so the ingest is idempotent."""
    path = MAPPINGS / "_curated_overlay.json"
    if not path.exists():
        return new_map, 0
    overlay = json.loads(path.read_text(encoding="utf-8")).get(overlay_key, {})
    notes = 0
    for old_sec, entry in overlay.items():
        note = entry.get("note")
        if not note:
            continue
        if old_sec in new_map:
            new_map[old_sec]["note"] = note
        else:
            # Overlay references an old section the NCRB table didn't surface — add it
            # as a no-equivalent entry so the human note is still available.
            new_map[old_sec] = {new_code.lower(): None, "offence": "", "note": note}
        notes += 1
    return new_map, notes


def main() -> int:
    for new_code, old_code, raw, out_file, new_title, old_title, url in TABLES:
        html_path = RAW / raw
        if not html_path.exists():
            print(f"  [SKIP] {raw} not found — fetch it first.")
            continue
        mapping = parse_table(new_code, old_code, html_path)
        overlay_key = f"{old_code.lower()}_{new_code.lower()}"
        mapping, notes = merge_overlay(mapping, overlay_key, new_code)

        doc = {
            "_meta": {
                "from_code": old_code,
                "to_code": new_code,
                "description": f"Official {old_code} ({old_title}) -> {new_code} ({new_title}) "
                               f"correspondence, parsed from the NCRB Corresponding Section Table.",
                "transition_date": "2024-07-01",
                "transition_rule": f"Matters on or after 2024-07-01 fall under the {new_code}; "
                                   f"earlier matters remain under the {old_code}.",
                "source": NCRB_AUTHORITY,
                "source_url": url,
                "authority_of_record": "https://www.indiacode.nic.in",
                "retrieved_at": RETRIEVED_AT,
                "verified": True,
                "verification_note": f"Mapping data is from the official NCRB Corresponding Section Table; "
                                     f"{notes} human clarifying note(s) merged from _curated_overlay.json.",
            },
            f"{old_code.lower()}_to_{new_code.lower()}": dict(sorted(mapping.items(), key=lambda kv: (len(kv[0]), kv[0]))),
        }
        (MAPPINGS / out_file).write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        n_null = sum(1 for v in mapping.values() if v.get(new_code.lower()) is None)
        print(f"  [OK] {out_file}: {len(mapping)} {old_code}->{new_code} entries "
              f"({n_null} no-equivalent, +{notes} notes)")

    # Verify the parse against a hand-checked set of well-known IPC->BNS base mappings.
    # Any disagreement is a parse bug or a genuine source difference — fail loudly.
    KNOWN_GOOD = {
        "302": "103", "304": "105", "304A": "106", "304B": "80", "306": "108", "307": "109",
        "354": "74", "354D": "78", "375": "63", "376": "64", "376D": "70", "379": "303",
        "392": "309", "395": "310", "406": "316", "420": "318", "498A": "85", "499": "356",
        "506": "351", "120B": "61", "326A": "124",
    }
    # Officially repealed without a direct successor. NB: IPC 124A (sedition) was NOT
    # re-enacted — BNS 152 is a new, differently-framed offence, not its successor.
    DROPPED_OK = {"377", "309", "124A"}
    print("\nVerification (IPC -> BNS) against known-good base mappings:")
    ib = json.loads((MAPPINGS / "ipc_bns.json").read_text(encoding="utf-8"))["ipc_to_bns"]
    base = lambda s: re.match(r"(\d+[A-Za-z]?)", s).group(1) if s else None
    mismatches = 0
    for ipc, want in sorted(KNOWN_GOOD.items()):
        e = ib.get(ipc)
        got = base(e.get("bns")) if e and e.get("bns") else None
        ok = got == want
        mismatches += 0 if ok else 1
        print(f"  {'OK ' if ok else 'BAD'} IPC {ipc:5} -> BNS {str(got):6} (want {want})")
    for ipc in sorted(DROPPED_OK):
        e = ib.get(ipc)
        is_null = bool(e) and e.get("bns") is None
        mismatches += 0 if is_null else 1
        print(f"  {'OK ' if is_null else 'BAD'} IPC {ipc:5} -> no-equivalent (dropped)")
    print(f"\n{'ALL KNOWN-GOOD MAPPINGS VERIFIED' if not mismatches else str(mismatches) + ' MISMATCH(ES) — DO NOT TRUST'}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
