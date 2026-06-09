"""Verify that citation source links actually resolve (HTTP 200).

Statute deep-links (BNS -> devgan.in) are the ones that can break — pages get
removed or renamed. Indian Kanoon *search* links (used for IPC and case Q&A) always
resolve, so we check every statute deep-link exhaustively and only sample the rest.

Run this after changing the corpus or the link logic — e.g. after dropping in the
official BNS bare act, or ingesting judgments with real document URLs.

Usage:
    python scripts/check_links.py            # all statute deep-links + a sample of search links
    python scripts/check_links.py --sample 25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.loaders import load_all  # noqa: E402

_UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120 Safari/537.36"
}


def _check(urls: list[str]) -> list[tuple[str, object]]:
    bad: list[tuple[str, object]] = []
    with httpx.Client(timeout=20, headers=_UA, follow_redirects=True) as client:
        for u in urls:
            try:
                code = client.get(u).status_code
                ok = code == 200
                print(f"  {'OK ' if ok else 'BAD'} {code}  {u}")
                if not ok:
                    bad.append((u, code))
            except Exception as e:  # noqa: BLE001
                print(f"  ERR      {u}  ({type(e).__name__})")
                bad.append((u, type(e).__name__))
    return bad


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=15, help="how many search-style links to spot-check")
    args = parser.parse_args()

    chunks = load_all()

    # Deep-links (devgan etc.) — check every unique one.
    deep, search, seen = [], [], set()
    for c in chunks:
        u = c.source_url()
        if not u or u in seen:
            continue
        seen.add(u)
        (search if "indiankanoon.org/search" in u else deep).append(u)

    print(f"\nDeep-links to verify exhaustively: {len(deep)}")
    bad = _check(sorted(deep))

    sample = sorted(search)[: args.sample]
    print(f"\nSearch-links (sample of {len(sample)} / {len(search)}):")
    bad += _check(sample)

    print("\n" + "=" * 60)
    if bad:
        print(f"{len(bad)} BROKEN LINK(S):")
        for u, s in bad:
            print(f"  [{s}] {u}")
        return 1
    print("ALL LINKS OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
