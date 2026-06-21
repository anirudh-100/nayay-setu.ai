"""Hindi answer smoke test — ask in Hindi, get a trustworthy Hindi answer.

Verifies the multilingual /ask path against a running backend:
  - a Hindi question yields a Hindi answer (Devanagari) with a Hindi disclaimer,
  - the law reference stays precise/standard (e.g. "BNS Section 103") and verified,
  - current-law bridging still fires (in Hindi),
  - an English question still answers in English (no regression),
  - a non-legal Hindi question abstains in Hindi.

Run the API first (uvicorn app.main:app), ideally with LLM_PROVIDER=claude (the local
3B model handles Hindi poorly). Usage: python scripts/hindi_smoke_test.py
"""
from __future__ import annotations

import os
import re
import sys

import httpx

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global _failures
    if not ok:
        _failures += 1
    line = f"  [{'PASS' if ok else 'FAIL'}] {name}"
    if detail:
        line += f"  - {detail}"
    print(line)


def ask(q: str, lang: str) -> dict:
    with httpx.Client(timeout=150) as c:
        r = c.post(f"{API_URL}/ask", json={"query": q, "language": lang})
        r.raise_for_status()
        return r.json()


def has_hindi(s: str) -> bool:
    return bool(_DEVANAGARI.search(s or ""))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        httpx.Client(timeout=8).get(f"{API_URL}/health").raise_for_status()
    except Exception as e:
        print(f"Backend not reachable at {API_URL} ({e}). Start it: uvicorn app.main:app")
        return 2

    print(f"Hindi smoke test @ {API_URL}\n" + "=" * 70)

    # 1. Hindi legal question -> Hindi answer, standard citation, verified.
    print("\n[1] Hindi: हत्या की सज़ा क्या है? (punishment for murder)")
    r = ask("हत्या की सज़ा क्या है?", "hi")
    print(f"    law_reference={r['law_reference']!r} verified={r['citation_verified']} conf={r['confidence']}")
    print(f"    answer: {r['answer'][:160]}")
    print(f"    note  : {r.get('current_law_note')}")
    check("answer is in Hindi", has_hindi(r["answer"]))
    check("law_reference is standard (BNS/section digits)", "103" in r["law_reference"] or "302" in r["law_reference"])
    check("citation verified", r["citation_verified"] is True)
    check("disclaimer is Hindi", has_hindi(r["disclaimer"]))
    check("current-law note is Hindi", r.get("current_law_note") and has_hindi(r["current_law_note"]))

    # 2. Hindi FIR (flagship current-law) -> BNSS 173 in Hindi.
    print("\n[2] Hindi: एफआईआर कैसे दर्ज करें? (how to file an FIR)")
    r = ask("एफआईआर कैसे दर्ज करें?", "hi")
    print(f"    law_reference={r['law_reference']!r} verified={r['citation_verified']} conf={r['confidence']}")
    print(f"    answer: {r['answer'][:160]}")
    check("answer is in Hindi", has_hindi(r["answer"]))
    check("leads with current BNSS 173 (or verified)", "173" in r["law_reference"] or r["citation_verified"])

    # 3. English still answers in English (no regression).
    print("\n[3] English: What is the punishment for murder?")
    r = ask("What is the punishment for murder?", "en")
    print(f"    law_reference={r['law_reference']!r} verified={r['citation_verified']}")
    check("English answer has NO Devanagari", not has_hindi(r["answer"]))
    check("English disclaimer (not Hindi)", not has_hindi(r["disclaimer"]))

    # 4. Non-legal Hindi -> abstain in Hindi.
    print("\n[4] Hindi non-legal: आज सोने का भाव क्या है? (price of gold)")
    r = ask("आज सोने का भाव क्या है?", "hi")
    print(f"    abstained={r['abstained']} conf={r['confidence']}")
    check("abstained", r["abstained"] is True)
    check("abstention message is Hindi", has_hindi(r["answer"]))

    print("=" * 70)
    print("ALL CHECKS PASSED" if _failures == 0 else f"{_failures} CHECK(S) FAILED")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
