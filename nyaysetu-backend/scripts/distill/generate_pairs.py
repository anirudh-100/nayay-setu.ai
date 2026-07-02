"""Teacher data generator for distillation: run the REAL pipeline, capture the pairs.

For each seed query in data/distill/seed_queries.jsonl (that isn't already in
data/distill/pairs.jsonl — resume by id), this script runs the production RAGService
end-to-end with the production teacher (claude-haiku-4-5 via get_llm()) and records:

  - the EXACT final prompt string sent to the teacher for the MAIN generation call
    (captured, not reimplemented: a RecordingLLM wrapper around get_llm() logs every
    (prompt, raw_dict) generate_json call the service makes — rewrites included),
  - the teacher's RAW parsed JSON reply for that call (the distillation target),
  - the final post-gate fields from the returned AskResponse (answer / law_reference /
    confidence / citation_verified / abstained),
  - the contract clean flag:
        clean = (not abstained) AND citation_verified
                AND final.answer == target_json.get("answer")
                AND confidence in ("high", "medium")
    anything else -> clean=false with a drop_reason.

The MAIN generation call is identified by marker: the prompt built from
rag_service.PROMPT_TEMPLATE (and ONLY that prompt) contains both the CONTEXT header
line and the literal '"law_reference"'; the rewrite/recovery prompts contain neither.
Both markers are asserted against the imported PROMPT_TEMPLATE at startup so a template
change fails loudly instead of mis-capturing.

Safety / ops:
  - --limit N        cap seeds generated this run (default 100 — the pilot),
  - --max-cost-usd   cumulative estimated-cost guard (default 3.0). Haiku pricing
                     estimate: in_tokens*1e-6*1.0 + out_tokens*1e-6*5.0, accumulated
                     over EVERY recorded LLM call (rewrites too, so the guard tracks
                     real spend); HARD STOP when exceeded,
  - progress print every 10 records (id, clean-rate so far, est cost so far),
  - per-seed exceptions are caught -> record clean=false, drop_reason="error: ...";
    the run never crashes,
  - abstained answers (abstention happens BEFORE the main generation call) -> record
    clean=false, drop_reason="abstained_pre_llm", prompt="", target_json={},
  - the pairs file is flushed after every record (kill-safe, resumable),
  - --report reads pairs.jsonl and prints clean-rate by kind/language + total
    estimated cost WITHOUT generating anything.

Teacher model: the contract teacher is claude-haiku-4-5. .env sets LLM_PROVIDER=claude
but not HIGH_POWER_MODEL (whose code default is an Opus tier), so this script pins
HIGH_POWER_MODEL=claude-haiku-4-5 in the process environment (env vars take precedence
over .env in pydantic-settings) BEFORE importing app modules. Override with
--teacher-model or a pre-set HIGH_POWER_MODEL env var.

Usage (from the backend root, venv python):
    .venv/Scripts/python.exe scripts/distill/generate_pairs.py --limit 100
    .venv/Scripts/python.exe scripts/distill/generate_pairs.py --report
    .venv/Scripts/python.exe scripts/distill/generate_pairs.py \
        --seeds data/distill/_smoke_seeds.jsonl --pairs data/distill/_smoke_pairs.jsonl --limit 3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:  # Windows consoles default to a legacy codepage; Hindi ids/reasons need UTF-8.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

DEFAULT_SEEDS = ROOT / "data" / "distill" / "seed_queries.jsonl"
DEFAULT_PAIRS = ROOT / "data" / "distill" / "pairs.jsonl"
DEFAULT_TEACHER = "claude-haiku-4-5"

# Main-generation-call markers. The prompt built from rag_service.PROMPT_TEMPLATE (and
# only that prompt) contains BOTH; the _standalone_query / _recovery_query rewrite
# prompts contain neither. Asserted against the imported template at startup.
MARKER_CONTEXT = "CONTEXT (each item is a retrieved source you may cite by its [LABEL]):"
MARKER_LAWREF = '"law_reference"'

# Haiku price estimate (USD per token), per the shared contract.
IN_USD_PER_TOKEN = 1e-6 * 1.0
OUT_USD_PER_TOKEN = 1e-6 * 5.0

CLEAN_CONFIDENCES = ("high", "medium")


def est_tokens(s: str) -> int:
    """Contract token estimate: ceil(chars / 4)."""
    return int(math.ceil(len(s or "") / 4))


# --------------------------------------------------------------------------- #
# Recording wrapper — capture, don't reimplement.
# --------------------------------------------------------------------------- #
class RecordingLLM:
    """Wraps the real LLM client; generate_json delegates and records every call."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.calls: list[tuple[str, dict]] = []  # (prompt, raw parsed dict), in order

    def generate_json(self, prompt: str) -> dict:
        raw = self._inner.generate_json(prompt)
        self.calls.append((prompt, raw))
        return raw

    def reset(self) -> None:
        self.calls = []

    def __getattr__(self, name):  # warmup() etc. pass through untouched
        return getattr(self._inner, name)


def find_main_calls(calls: list[tuple[str, dict]]) -> list[int]:
    """Indices of recorded calls whose prompt carries BOTH main-call markers."""
    return [i for i, (p, _) in enumerate(calls) if MARKER_CONTEXT in p and MARKER_LAWREF in p]


def calls_cost_usd(calls: list[tuple[str, dict]]) -> float:
    """Estimated Haiku spend for a list of recorded calls (rewrites included)."""
    cost = 0.0
    for prompt, raw in calls:
        out_chars = json.dumps(raw, ensure_ascii=False) if raw else ""
        cost += est_tokens(prompt) * IN_USD_PER_TOKEN + est_tokens(out_chars) * OUT_USD_PER_TOKEN
    return cost


# --------------------------------------------------------------------------- #
# JSONL helpers
# --------------------------------------------------------------------------- #
def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ! skipping malformed line {n} in {path.name}: {e}")
    return rows


def existing_ids(pairs_path: Path) -> set[str]:
    if not pairs_path.exists():
        return set()
    return {r.get("id") for r in read_jsonl(pairs_path) if r.get("id")}


# --------------------------------------------------------------------------- #
# Record construction (the shared pairs.jsonl contract)
# --------------------------------------------------------------------------- #
def build_record(seed: dict, resp, calls: list[tuple[str, dict]]) -> dict:
    """One contract-conformant pairs record from a seed + AskResponse + recorded calls."""
    final = {
        "answer": resp.answer,
        "law_reference": resp.law_reference,
        "confidence": resp.confidence,
        "citation_verified": bool(resp.citation_verified),
        "abstained": bool(resp.abstained),
    }

    if resp.abstained:
        # Abstention fires on weak retrieval BEFORE the main generation call, so there is
        # (usually) no main call to capture. Never a training pair.
        prompt, target_json = "", {}
        clean, drop_reason = False, "abstained_pre_llm"
    else:
        mains = find_main_calls(calls)
        if not mains:
            prompt, target_json = "", {}
            clean, drop_reason = False, "main_call_not_found"
        else:
            if len(mains) > 1:  # should never happen — markers are template-unique
                print(f"  ! WARNING: {len(mains)} main-marker calls for id={seed.get('id')}; using the last")
            prompt, target_json = calls[mains[-1]]
            reasons: list[str] = []
            if not final["citation_verified"]:
                reasons.append("citation_unverified")
            if final["answer"] != target_json.get("answer"):
                reasons.append("answer_mutated_by_postprocess")
            if final["confidence"] not in CLEAN_CONFIDENCES:
                reasons.append(f"confidence_{final['confidence']}")
            clean = not reasons
            drop_reason = "; ".join(reasons) if reasons else None

    # est_tokens sums ALL teacher calls for this seed (rewrites + recovery + main), not
    # just the main call — Gate-1's cost projection and the $25 trim rule consume this
    # number, so under-counting the rewrite overhead (~1-2 extra calls on most narrative
    # seeds) would systematically under-project the full-run spend.
    tok_in = sum(est_tokens(p) for p, _ in calls)
    tok_out = sum(est_tokens(json.dumps(r, ensure_ascii=False)) for _, r in calls)
    return {
        "id": seed["id"],
        "domain": seed.get("domain", ""),
        "kind": seed.get("kind", ""),
        "language": seed.get("language", "en"),
        "query": seed.get("query", ""),
        "history": seed.get("history") or [],
        "prompt": prompt,
        "target_json": target_json,
        "final": final,
        "clean": clean,
        "drop_reason": drop_reason,
        "est_tokens": {"in": tok_in, "out": tok_out},
    }


def error_record(seed: dict, exc: Exception) -> dict:
    return {
        "id": seed.get("id", ""),
        "domain": seed.get("domain", ""),
        "kind": seed.get("kind", ""),
        "language": seed.get("language", "en"),
        "query": seed.get("query", ""),
        "history": seed.get("history") or [],
        "prompt": "",
        "target_json": {},
        "final": {
            "answer": "",
            "law_reference": "",
            "confidence": "",
            "citation_verified": False,
            "abstained": False,
        },
        "clean": False,
        "drop_reason": f"error: {type(exc).__name__}: {exc}"[:500],
        "est_tokens": {"in": 0, "out": 0},
    }


# --------------------------------------------------------------------------- #
# --report mode (no generation, no LLM)
# --------------------------------------------------------------------------- #
def report(pairs_path: Path) -> int:
    if not pairs_path.exists():
        print(f"No pairs file at {pairs_path}")
        return 2
    rows = read_jsonl(pairs_path)
    if not rows:
        print(f"{pairs_path} is empty.")
        return 0

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r.get("kind", "?"), r.get("language", "?"))].append(r)

    total_in = sum(int(r.get("est_tokens", {}).get("in", 0)) for r in rows)
    total_out = sum(int(r.get("est_tokens", {}).get("out", 0)) for r in rows)
    total_cost = total_in * IN_USD_PER_TOKEN + total_out * OUT_USD_PER_TOKEN
    n_clean = sum(1 for r in rows if r.get("clean"))

    print(f"pairs report: {pairs_path}")
    print("=" * 68)
    print(f"{'kind':<14} {'lang':<5} {'n':>5} {'clean':>6} {'rate':>7}")
    print("-" * 68)
    for (kind, lang) in sorted(groups):
        g = groups[(kind, lang)]
        c = sum(1 for r in g if r.get("clean"))
        print(f"{kind:<14} {lang:<5} {len(g):>5} {c:>6} {c / len(g):>6.1%}")
    print("-" * 68)
    print(f"{'TOTAL':<14} {'':<5} {len(rows):>5} {n_clean:>6} {n_clean / len(rows):>6.1%}")

    drops = Counter((r.get("drop_reason") or "").split(";")[0].strip() for r in rows if not r.get("clean"))
    if drops:
        print("\ndrop reasons:")
        for reason, n in drops.most_common():
            print(f"  {n:>5}  {reason}")

    print(f"\nest tokens (ALL teacher calls incl. rewrites): in={total_in:,} out={total_out:,}")
    print(f"est cost   (ALL teacher calls incl. rewrites): ${total_cost:.4f}  "
          f"(Haiku @ $1.0/M in, $5.0/M out)")
    return 0


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def generate(args: argparse.Namespace) -> int:
    # Pin the contract teacher BEFORE importing app modules (env var beats .env).
    os.environ.setdefault("HIGH_POWER_MODEL", args.teacher_model)

    from app.config import settings  # noqa: E402  (heavy imports after env pin)
    from app.schemas.ask import ConversationTurn  # noqa: E402
    from app.services.llm_service import get_llm  # noqa: E402
    from app.services.rag_service import PROMPT_TEMPLATE, RAGService  # noqa: E402

    # Fail loudly if the template drifted away from our markers (mis-capture guard).
    assert MARKER_CONTEXT in PROMPT_TEMPLATE, "MARKER_CONTEXT no longer in PROMPT_TEMPLATE"
    assert MARKER_LAWREF in PROMPT_TEMPLATE, "MARKER_LAWREF no longer in PROMPT_TEMPLATE"

    if settings.llm_provider.strip().lower() != "claude":
        print(f"ERROR: llm_provider={settings.llm_provider!r} — the teacher must be Claude "
              f"(set LLM_PROVIDER=claude in .env).")
        return 2
    # HARD teacher assert: setdefault() above loses to a pre-set HIGH_POWER_MODEL in the
    # shell (e.g. a stray claude-opus-4-8), which would silently bill 5x off-contract for
    # the whole run. Abort loudly on any mismatch instead of trusting the default.
    if settings.high_power_model != args.teacher_model:
        print(f"ERROR: resolved teacher model is {settings.high_power_model!r} but the run "
              f"contract says {args.teacher_model!r}. A pre-set HIGH_POWER_MODEL env var is "
              f"overriding the pin — unset it (or pass a matching --teacher-model) and rerun.")
        return 2

    seeds_path = Path(args.seeds)
    pairs_path = Path(args.pairs)
    if not seeds_path.exists():
        print(f"ERROR: seeds file not found: {seeds_path}")
        return 2
    pairs_path.parent.mkdir(parents=True, exist_ok=True)

    seeds = read_jsonl(seeds_path)
    done_ids = existing_ids(pairs_path)
    todo = [s for s in seeds if s.get("id") and s["id"] not in done_ids]
    print(f"Teacher: {settings.llm_provider} ({settings.high_power_model})")
    print(f"Seeds: {len(seeds)} total | {len(done_ids)} already in {pairs_path.name} "
          f"| {len(todo)} to do | limit={args.limit} | max-cost=${args.max_cost_usd:.2f}")
    if not todo:
        print("Nothing to do — all seeds already generated (resume found no gap).")
        return 0

    print("Loading pipeline (retriever models + index)…")
    recorder = RecordingLLM(get_llm())
    svc = RAGService(llm=recorder)

    cum_cost = 0.0
    generated = 0
    clean_count = 0
    stopped_for_cost = False

    with pairs_path.open("a", encoding="utf-8") as out:
        for seed in todo[: args.limit]:
            if cum_cost >= args.max_cost_usd:
                stopped_for_cost = True
                break

            recorder.reset()
            try:
                history = [ConversationTurn(**t) for t in (seed.get("history") or [])]
                resp = svc.answer(seed["query"], language=seed.get("language", "en"), history=history)
                record = build_record(seed, resp, recorder.calls)
            except Exception as exc:  # never crash the run
                print(f"  ! error on id={seed.get('id')}: {type(exc).__name__}: {exc}")
                record = error_record(seed, exc)

            cum_cost += calls_cost_usd(recorder.calls)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            os.fsync(out.fileno())

            generated += 1
            clean_count += 1 if record["clean"] else 0
            if generated % 10 == 0 or generated == 1:
                print(f"[{generated}/{min(len(todo), args.limit)}] id={record['id']} "
                      f"clean-rate={clean_count / generated:.0%} est-cost=${cum_cost:.4f}")

    print("=" * 68)
    if stopped_for_cost:
        print(f"HARD STOP: estimated cumulative cost ${cum_cost:.4f} reached "
              f"--max-cost-usd {args.max_cost_usd:.2f} — run halted (resume later; done work is saved).")
    rate = f"{clean_count / generated:.0%}" if generated else "n/a"
    print(f"Done: {generated} record(s) appended to {pairs_path} | clean {clean_count}/{generated} "
          f"({rate}) | est cost this run ${cum_cost:.4f} (all LLM calls incl. rewrites)")
    return 3 if stopped_for_cost else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Distillation teacher-pair generator (real pipeline, recorded).")
    ap.add_argument("--seeds", default=str(DEFAULT_SEEDS), help="seed_queries.jsonl path")
    ap.add_argument("--pairs", default=str(DEFAULT_PAIRS), help="pairs.jsonl output path (append/resume)")
    ap.add_argument("--limit", type=int, default=100, help="max seeds to generate this run (pilot=100)")
    ap.add_argument("--max-cost-usd", type=float, default=3.0, help="hard stop when est cost exceeds this")
    ap.add_argument("--teacher-model", default=DEFAULT_TEACHER,
                    help="pins HIGH_POWER_MODEL unless already set in the environment")
    ap.add_argument("--report", action="store_true", help="print clean-rate/cost report; no generation")
    args = ap.parse_args()

    if args.report:
        return report(Path(args.pairs))
    return generate(args)


if __name__ == "__main__":
    raise SystemExit(main())
