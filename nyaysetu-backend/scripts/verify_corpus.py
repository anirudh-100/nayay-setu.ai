"""Corpus trust report — how much of the knowledge base is actually verified.

The product's promise is that every answer is grounded in a source the user can
trust. This script makes that measurable: it loads every chunk the engine would
index and reports, per act/source, how many are **official** (text taken from and
checked against an authority), **curated** (hand-compiled starters), or
**unverified** (bulk-ingested datasets). It also runs consistency checks on
``data/acts/registry.json`` so a source can't silently claim to be official without
provenance to back it.

Run this whenever you add or promote a source — it tells you exactly how far the
corpus is from "fully official", which is the moat this project is built on.

Usage:
    python scripts/verify_corpus.py
    python scripts/verify_corpus.py --strict   # exit non-zero if any check fails

No models, no Ollama, no built index needed — it only parses the data files.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.rag.loaders import load_all  # noqa: E402

_LEVELS = ("official", "curated", "unverified")


def _registry_path() -> Path:
    return Path(settings.data_dir) / "acts" / "registry.json"


def _check_registry() -> list[str]:
    """Provenance consistency checks. Returns a list of problem strings (empty == OK)."""
    problems: list[str] = []
    path = _registry_path()
    if not path.exists():
        return [f"registry not found at {path}"]

    acts = json.loads(path.read_text(encoding="utf-8")).get("acts", [])
    for act in acts:
        code = act.get("code", "?")
        prov = act.get("provenance")
        if not prov:
            problems.append(f"{code}: missing 'provenance' block")
            continue
        if not prov.get("authority"):
            problems.append(f"{code}: provenance.authority is empty")
        # An act may only claim to be verified/official if it records WHEN its text
        # was taken from the authority — otherwise "official" is unprovable.
        if act.get("verified") and not prov.get("retrieved_at"):
            problems.append(
                f"{code}: verified:true but provenance.retrieved_at is null "
                f"(cannot prove the text matches the authority)"
            )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="exit 1 if any consistency check fails")
    args = parser.parse_args()

    chunks = load_all()

    # Group by a friendly key: act code for statutes, else the source type.
    by_group: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    authority: dict[str, str] = {}
    status: dict[str, str] = {}
    totals: dict[str, int] = defaultdict(int)

    for c in chunks:
        key = c.act or c.source_type
        by_group[key][c.verification] += 1
        totals[c.verification] += 1
        if c.source_authority:
            authority.setdefault(key, c.source_authority)
        if c.act and c.code_status != "unknown":
            status.setdefault(key, c.code_status)

    total = sum(totals.values()) or 1

    print("\nCorpus trust report")
    print("=" * 72)
    print(f"  {'SOURCE':<14}{'STATUS':<10}{'OFFICIAL':>9}{'CURATED':>9}{'UNVERIF':>9}   AUTHORITY")
    print("  " + "-" * 68)
    for key in sorted(by_group):
        g = by_group[key]
        print(
            f"  {key:<14}{status.get(key, ''):<10}"
            f"{g.get('official', 0):>9}{g.get('curated', 0):>9}{g.get('unverified', 0):>9}"
            f"   {authority.get(key, '')[:30]}"
        )

    print("  " + "-" * 68)
    print(
        f"  {'TOTAL':<14}{'':<10}"
        f"{totals.get('official', 0):>9}{totals.get('curated', 0):>9}{totals.get('unverified', 0):>9}"
    )
    pct = 100.0 * totals.get("official", 0) / total
    print(f"\n  Official-source coverage: {pct:.1f}%  ({totals.get('official', 0)}/{total} chunks)")

    print("\nConsistency checks (registry provenance):")
    problems = _check_registry()
    if not problems:
        print("  [OK] every act has provenance; no act claims 'official' without a retrieval date.")
    else:
        for p in problems:
            print(f"  [FAIL] {p}")

    print("=" * 72)
    if problems and args.strict:
        print(f"{len(problems)} consistency problem(s) — failing (--strict).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
