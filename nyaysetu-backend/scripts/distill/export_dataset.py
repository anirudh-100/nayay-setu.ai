"""Export the distillation fine-tuning dataset from teacher pairs.

Reads  : data/distill/pairs.jsonl   (one JSON per line, produced by the pair generator)
Writes : data/distill/train.jsonl   (Unsloth chat format, ~95% of clean pairs)
         data/distill/val.jsonl     (~5% of clean pairs, deterministic by id hash)
         data/distill/dataset_stats.md

Shared interface contract (must match the pair generator and the training notebook):
  - Keep ONLY pairs with clean == true.
  - Output record shape (one JSON object per line):
      {"messages": [{"role": "user",      "content": <prompt>},
                    {"role": "assistant", "content": json.dumps(target_json, ensure_ascii=False)}]}
  - 95/5 split, DETERMINISTIC by id hash: a pair goes to val when
      int(sha1(id).hexdigest(), 16) % 100 < 5
    Stable across runs/machines; no RNG involved, so re-running after the pair
    generator appends more rows only ever ADDS records to each side.

Graceful behaviour on tiny/empty inputs (the pair generator may have produced only a
small smoke file, or nothing yet):
  - pairs.jsonl missing  -> print a message and exit 0 (write nothing).
  - zero clean pairs     -> write empty train/val plus a stats file that says why.
  - val bucket empty but >= 2 clean pairs -> deterministically promote the clean pair
    with the SMALLEST sha1(id) into val, so the notebook's inference sanity-check cell
    always has a validation example. (Still deterministic: same input -> same split.)

This script never makes an LLM call. Idempotent: outputs are fully rewritten each run.

Run (Git Bash on Windows, from the backend root):
  export PYTHONIOENCODING=utf-8 && .venv/Scripts/python.exe scripts/distill/export_dataset.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------- #
# Teacher pricing — claude-haiku-4-5 (Anthropic list price, USD per 1M tokens).
# Used only for the *estimated* spend line in dataset_stats.md. Token counts in
# pairs.jsonl are themselves estimates (ceil(chars / 4)), so treat the figure as
# an order-of-magnitude sanity number, not an invoice.
# --------------------------------------------------------------------------- #
PRICE_IN_PER_MTOK = 1.00
PRICE_OUT_PER_MTOK = 5.00

VAL_PERCENT = 5          # ~5% of clean pairs go to val (bucket 0..4 of 100)
MAX_SEQ_LEN = 8192       # the notebook trains with this context; flag pairs above it

DEFAULT_PAIRS = Path("data/distill/pairs.jsonl")
DEFAULT_OUT_DIR = Path("data/distill")


def _id_bucket(pair_id: str) -> int:
    """Deterministic 0..99 bucket from the pair id (sha1, no RNG)."""
    return int(hashlib.sha1(pair_id.encode("utf-8")).hexdigest(), 16) % 100


def _sort_key(pair_id: str) -> str:
    """Stable ordering key used for the tiny-input val promotion."""
    return hashlib.sha1(pair_id.encode("utf-8")).hexdigest()


def _est_tokens(pair: dict) -> tuple[int, int]:
    """(input, output) token estimates for a pair; falls back to ceil(chars/4)."""
    est = pair.get("est_tokens") or {}
    tin = est.get("in")
    tout = est.get("out")
    if not isinstance(tin, int):
        tin = math.ceil(len(str(pair.get("prompt", ""))) / 4)
    if not isinstance(tout, int):
        tout = math.ceil(len(json.dumps(pair.get("target_json", {}), ensure_ascii=False)) / 4)
    return tin, tout


def _to_chat_record(pair: dict) -> dict:
    """Contract item 3: the exact Unsloth chat-format record the notebook expects."""
    return {
        "messages": [
            {"role": "user", "content": pair["prompt"]},
            {"role": "assistant", "content": json.dumps(pair["target_json"], ensure_ascii=False)},
        ]
    }


def _fmt_counter(counter: Counter, total: int) -> str:
    """Markdown table body for a Counter, most-common first."""
    if not counter:
        return "| (none) | 0 | - |\n"
    lines = []
    for key, n in counter.most_common():
        pct = (100.0 * n / total) if total else 0.0
        lines.append(f"| {key} | {n} | {pct:.1f}% |")
    return "\n".join(lines) + "\n"


def _median_or_zero(values: list[int]) -> float:
    return float(statistics.median(values)) if values else 0.0


def export(pairs_path: Path, out_dir: Path) -> int:
    if not pairs_path.exists():
        print(f"[export_dataset] {pairs_path} not found - nothing to export yet.")
        print("[export_dataset] Run the pair generator first, then re-run this script.")
        return 0

    # ------------------------------------------------------------------ #
    # Parse pairs.jsonl (append-only file; tolerate blank/malformed lines).
    # ------------------------------------------------------------------ #
    pairs: list[dict] = []
    malformed = 0
    seen_ids: set[str] = set()
    duplicates = 0
    with pairs_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(obj, dict) or not obj.get("id"):
                malformed += 1
                continue
            # Resumable/append-only file: keep the LAST record per id.
            if obj["id"] in seen_ids:
                duplicates += 1
                pairs = [p for p in pairs if p["id"] != obj["id"]]
            seen_ids.add(obj["id"])
            pairs.append(obj)

    clean = [p for p in pairs if p.get("clean") is True and p.get("prompt") and isinstance(p.get("target_json"), dict)]
    clean_ids = {p["id"] for p in clean}
    dropped = [p for p in pairs if p["id"] not in clean_ids]

    # ------------------------------------------------------------------ #
    # Deterministic 95/5 split by id hash.
    # ------------------------------------------------------------------ #
    train = [p for p in clean if _id_bucket(p["id"]) >= VAL_PERCENT]
    val = [p for p in clean if _id_bucket(p["id"]) < VAL_PERCENT]

    promoted_note = ""
    if not val and len(clean) >= 2:
        # Tiny-input safeguard (e.g. a 3-record smoke file where no id hashed into
        # the 5% bucket): promote ONE clean pair - the one with the smallest
        # sha1(id) - so the notebook's val-based sanity check never breaks.
        # Deterministic: same input file always promotes the same pair.
        promote = min(train, key=lambda p: _sort_key(p["id"]))
        train = [p for p in train if p["id"] != promote["id"]]
        val = [promote]
        promoted_note = (
            f"(val bucket was empty for this tiny input; pair `{promote['id']}` was "
            "deterministically promoted to val so the notebook has a validation example)"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    stats_path = out_dir / "dataset_stats.md"

    for path, subset in ((train_path, train), (val_path, val)):
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for p in subset:
                fh.write(json.dumps(_to_chat_record(p), ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ #
    # Stats.
    # ------------------------------------------------------------------ #
    def counters(subset: list[dict]) -> tuple[Counter, Counter, Counter]:
        by_kind = Counter(p.get("kind", "?") for p in subset)
        by_lang = Counter(p.get("language", "?") for p in subset)
        by_conf = Counter((p.get("final") or {}).get("confidence", "?") for p in subset)
        return by_kind, by_lang, by_conf

    all_kind, all_lang, all_conf = counters(pairs)
    cl_kind, cl_lang, cl_conf = counters(clean)
    drop_reasons = Counter(str(p.get("drop_reason") or "unspecified") for p in dropped)

    # Cost estimate covers ALL parsed pairs - dropped pairs also cost real teacher money.
    total_in = sum(_est_tokens(p)[0] for p in pairs)
    total_out = sum(_est_tokens(p)[1] for p in pairs)
    est_cost = (total_in / 1_000_000) * PRICE_IN_PER_MTOK + (total_out / 1_000_000) * PRICE_OUT_PER_MTOK

    # Sequence-length stats over the CLEAN pairs (what actually trains). The
    # notebook's max_seq_len must fit prompt + target + chat-template overhead.
    clean_in = [_est_tokens(p)[0] for p in clean]
    clean_out = [_est_tokens(p)[1] for p in clean]
    clean_total = [i + o for i, o in zip(clean_in, clean_out)]
    over_limit = sum(1 for t in clean_total if t > MAX_SEQ_LEN)
    near_limit = sum(1 for t in clean_total if MAX_SEQ_LEN * 0.9 < t <= MAX_SEQ_LEN)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md: list[str] = []
    md.append("# Distillation dataset stats\n")
    md.append(f"_Generated {now} by `scripts/distill/export_dataset.py` from `{pairs_path.as_posix()}`._\n")
    md.append("## Input\n")
    md.append(f"- Parsed pairs: **{len(pairs)}** (malformed lines skipped: {malformed}, duplicate ids superseded: {duplicates})")
    md.append(f"- Clean (`clean == true`): **{len(clean)}**")
    md.append(f"- Dropped: **{len(dropped)}**\n")
    md.append("## Split (95/5 deterministic by sha1(id) % 100 < 5)\n")
    md.append(f"- `train.jsonl`: **{len(train)}** records")
    md.append(f"- `val.jsonl`: **{len(val)}** records {promoted_note}\n")

    md.append("## Counts by kind\n")
    md.append("| kind | all pairs | share |\n|---|---|---|")
    md.append(_fmt_counter(all_kind, len(pairs)))
    md.append("| kind (clean only) | clean pairs | share |\n|---|---|---|")
    md.append(_fmt_counter(cl_kind, len(clean)))

    md.append("## Counts by language\n")
    md.append("| language | all pairs | share |\n|---|---|---|")
    md.append(_fmt_counter(all_lang, len(pairs)))
    md.append("| language (clean only) | clean pairs | share |\n|---|---|---|")
    md.append(_fmt_counter(cl_lang, len(clean)))

    md.append("## Counts by confidence (final.confidence)\n")
    md.append("| confidence | all pairs | share |\n|---|---|---|")
    md.append(_fmt_counter(all_conf, len(pairs)))
    md.append("| confidence (clean only) | clean pairs | share |\n|---|---|---|")
    md.append(_fmt_counter(cl_conf, len(clean)))

    md.append("## Drop-reason histogram\n")
    md.append("| drop_reason | dropped pairs | share of dropped |\n|---|---|---|")
    md.append(_fmt_counter(drop_reasons, len(dropped)))

    md.append("## Estimated teacher cost (claude-haiku-4-5)\n")
    md.append(f"- Input tokens (est, all pairs incl. dropped): **{total_in:,}**")
    md.append(f"- Output tokens (est, all pairs incl. dropped): **{total_out:,}**")
    md.append(
        f"- Estimated spend at ${PRICE_IN_PER_MTOK:.2f}/MTok in + ${PRICE_OUT_PER_MTOK:.2f}/MTok out: "
        f"**${est_cost:.4f}**"
    )
    md.append("- Token counts are `ceil(chars/4)` estimates from the pair generator, not tokenizer-exact.\n")

    md.append("## Prompt/sequence length (clean pairs) - IMPORTANT for the notebook's max_seq_len\n")
    md.append(f"- Prompt tokens (est_tokens.in): max **{max(clean_in) if clean_in else 0:,}**, "
              f"median **{_median_or_zero(clean_in):,.0f}**")
    md.append(f"- Target tokens (est_tokens.out): max **{max(clean_out) if clean_out else 0:,}**, "
              f"median **{_median_or_zero(clean_out):,.0f}**")
    md.append(f"- Full sequence (prompt + target): max **{max(clean_total) if clean_total else 0:,}**, "
              f"median **{_median_or_zero(clean_total):,.0f}**")
    md.append(f"- Clean pairs whose est. full sequence EXCEEDS max_seq_len={MAX_SEQ_LEN}: **{over_limit}** "
              "(these would be truncated in training - if > 0, consider dropping or raising max_seq_len)")
    md.append(f"- Clean pairs within 10% of the limit ({int(MAX_SEQ_LEN * 0.9)}-{MAX_SEQ_LEN} est. tokens): "
              f"**{near_limit}** (char/4 underestimates real tokenizers on legal text - keep headroom)\n")

    if not clean:
        md.append("> **No clean pairs yet.** train.jsonl / val.jsonl were written empty. "
                  "Re-run after the pair generator has produced clean pairs.\n")

    with stats_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(md))

    print(f"[export_dataset] pairs={len(pairs)} clean={len(clean)} dropped={len(dropped)} "
          f"-> train={len(train)} val={len(val)}")
    print(f"[export_dataset] wrote {train_path}, {val_path}, {stats_path}")
    if over_limit:
        print(f"[export_dataset] WARNING: {over_limit} clean pair(s) exceed max_seq_len={MAX_SEQ_LEN} (est).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS,
                        help=f"input pairs.jsonl (default: {DEFAULT_PAIRS})")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"output directory for train/val/stats (default: {DEFAULT_OUT_DIR})")
    args = parser.parse_args()
    return export(args.pairs, args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
