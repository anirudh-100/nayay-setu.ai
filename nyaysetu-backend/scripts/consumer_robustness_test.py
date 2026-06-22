"""Robustness test for lay-phrased consumer (and any civil) retrieval.

A raw lay consumer narrative ("the shopkeeper sold me a defective mixer and refuses
to refund") reranks deeply negative; the LLM query-rewrite in
RAGService._standalone_query() is what rescues it, but it is NON-DETERMINISTIC — some
samplings produce a strong statute-matching query (answers), some a weak one (top
rerank < 0 -> the engine abstains). Live, the 6 consumer queries flaked to ~3/6.

This exercises the robustness fix (RAGService._robust_retrieve): when the rewrite
retrieval lands below the (correctly calibrated) abstain threshold, retry once with a
fresh rewrite AND the original text, then keep the best-scoring merged set. We run the
full answer() path 3x per query and count how many runs ANSWER (not abstain).

PASS criteria:
  - consumer queries: answer consistently (>= 3/3 each, i.e. no flakiness),
  - criminal queries: still answer,
  - junk queries: still abstain on ALL runs (threshold not weakened).

    $env:PYTHONIOENCODING="utf-8"; .venv/Scripts/python.exe scripts/consumer_robustness_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.services.rag_service import RAGService  # noqa: E402

RUNS = 3

# Six lay-phrased CONSUMER PROTECTION matters (defective goods / deficient service /
# refund / e-commerce) — the area the ingested Consumer Protection Act, 2019 covers. Each
# is a described situation that names NO statute, so it reranks deeply negative verbatim
# and depends on the (non-deterministic) rewrite — exactly the flakiness we are fixing.
CONSUMER = [
    "the shopkeeper sold me a defective mixer and refuses to refund",
    "I bought a new fridge online and it arrived broken, the seller won't replace it",
    "the showroom delivered a damaged sofa and is refusing to take it back or refund",
    "the mobile company keeps charging me for a plan I never agreed to and won't stop",
    "a courier company lost my parcel and is refusing to compensate me for it",
    "the car service centre damaged my vehicle and is denying any responsibility",
]
CRIMINAL = [
    "knife attack ICU",
    "online cheating 50000",
]
JUNK = [
    "which smartphone should I buy with best camera and battery",
    "good pizza recipe with cheese",
]


def run_group(svc, name, queries):
    print(f"\n=== {name} ===")
    rows = []
    for q in queries:
        answered = 0
        scores = []
        for _ in range(RUNS):
            try:
                resp = svc.answer(q)
                if not resp.abstained:
                    answered += 1
                # surface the realized top-of-context strength via citations presence
                scores.append("ans" if not resp.abstained else "abstain")
            except Exception as e:
                scores.append(f"ERR:{e}")
        rows.append((q, answered))
        print(f"  [{answered}/{RUNS}] {q[:70]}")
    return rows


def main():
    svc = RAGService()

    consumer_rows = run_group(svc, "CONSUMER (must answer consistently)", CONSUMER)
    criminal_rows = run_group(svc, "CRIMINAL (must still answer)", CRIMINAL)
    junk_rows = run_group(svc, "JUNK (must abstain ALL runs)", JUNK)

    consumer_answered = sum(a for _, a in consumer_rows)
    consumer_total = len(CONSUMER) * RUNS
    consumer_full = sum(1 for _, a in consumer_rows if a == RUNS)

    criminal_answered = sum(a for _, a in criminal_rows)
    criminal_total = len(CRIMINAL) * RUNS

    junk_answered = sum(a for _, a in junk_rows)

    print("\n--- SUMMARY ---")
    print(f"consumer answered runs: {consumer_answered}/{consumer_total} "
          f"({consumer_full}/{len(CONSUMER)} queries at full {RUNS}/{RUNS})")
    print(f"criminal answered runs: {criminal_answered}/{criminal_total}")
    print(f"junk answered runs:     {junk_answered}/{len(JUNK) * RUNS} (want 0)")

    # PASS gates. Consumer must be robust: every consumer query answers on EVERY run.
    consumer_ok = consumer_full == len(CONSUMER)
    criminal_ok = criminal_answered == criminal_total
    junk_ok = junk_answered == 0

    print("\n--- GATES ---")
    print(f"[{'PASS' if consumer_ok else 'FAIL'}] consumer answers consistently "
          f"({consumer_full}/{len(CONSUMER)} at full {RUNS}/{RUNS})")
    print(f"[{'PASS' if criminal_ok else 'FAIL'}] criminal still answers "
          f"({criminal_answered}/{criminal_total})")
    print(f"[{'PASS' if junk_ok else 'FAIL'}] junk abstains on all runs "
          f"({junk_answered} answered, want 0)")

    ok = consumer_ok and criminal_ok and junk_ok
    print(f"\n{'ALL GATES PASS' if ok else 'SOME GATES FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
