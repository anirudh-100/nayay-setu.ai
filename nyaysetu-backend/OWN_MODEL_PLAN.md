# NyaySetu Own-Model Plan — the fine-tuned Indian legal engine (v2)

> **Status: ACTIVE.** v2 supersedes v1 entirely. Built from: the implemented pipeline
> (committed `a2675e8`), a 4-stream deep-research pass (benchmarks/competitors, data,
> training, serving — all claims source-checked), an adversarial critique of that
> research, and a **completed, measured 100-seed Gate-1 pilot**. Numbers in this doc
> marked *measured* are real; everything else carries its uncertainty.

---

## 0. The claim we are building toward (honest version)

"Beat Indian law at the very best level" is not a falsifiable engineering goal — frontier
models already beat human exam toppers on Indian legal MCQs (Gemini 2.5 Pro +7.6..+25.3
pts on the adalat-ai court-ready study), and no 4B model approaches that closed-book. What
IS winnable — and currently **unclaimed by anyone** — is this, on the public model card:

1. **SYSTEM claim** — *"NyaySetu scores X% on AIBE past papers and Y% on the adalat-ai
   6,218-MCQ benchmark, retrieval-on, abstentions counted as wrong — beating every
   published open Indian legal fine-tune."* The bar is low and quantified: OpenNyAI's
   **Aalap (Mistral-7B) scores 25.56% on AIBE** (pass mark 40%, gpt-3.5-turbo 58.72%).
   InLegalLLaMA is a task model; SaulLM is US/EU law. **No open Indian legal model
   ≤8B publishes a credible assistant-grade scorecard.**
2. **TRUST claim (the real moat)** — *the only Indian legal AI publishing hallucination
   metrics*: hallucinated-citation rate, abstention precision/recall, current-law-bridge
   accuracy, Hindi/English parity, red-team pass rate. No competitor (Aalap, Lexlegis,
   CaseMine, Jugalbandi) publishes ANY of these.
3. **MODEL claim (only if measured)** — base-Qwen3-4B vs fine-tuned delta on the same
   harness, proving the distillation itself added value.

Timing pressure: BharatGen just shipped BhashaBench-Legal (24k validated legal MCQs) —
that is release-grade eval infrastructure, i.e. a funded Indian legal model is likely
coming. **First public scorecard + first published trust metrics wins the narrative.**

---

## 1. Locked decisions & live status

| Decision | Value | Why |
|---|---|---|
| Student | **Qwen3-4B Instruct (2507)** | Apache 2.0. NOT Qwen2.5-3B: research/non-commercial license. 8B rung later (§5 v3). |
| Teacher | **claude-haiku-4-5** | The audited production engine — the student clones validated behavior. $1/$5 per MTok. |
| Method | QLoRA SFT (Unsloth) → GGUF q4_K_M → Ollama `nyaysetu-legal` | Free-tier trainable; laptop-serveable. |
| Training venue | **Kaggle primary** (verified 30 GPU-h/wk, 12h sessions), Colab fallback only | Free Colab 2026: dynamic ~15–30 h/wk cap, 90-min idle kill — unfit as primary. |
| Data | Trust-gated pairs from the REAL RAGService pipeline only | Train ONLY on gate-clean outputs. Knowledge stays in the retrieval corpus; the student learns *behavior*. |

**Live status (2026-06-24):**
- Pipeline implemented, verified, committed (`a2675e8`): `scripts/distill/` (seeds →
  pairs → export → eval) + `training/` (Colab notebook + Modelfile).
- **Gate-1 pilot DONE (measured):** 100 stratified seeds, **$0.6509** all-in.
  Clean-keep by kind: followup **10/10**, narrative_en **21/25 (84%)**, narrative_hi
  **12/20 (60%)**, indicqa **19/40 (47.5%)**, thin 1/5 (by design — they hedge/abstain).
  **Total 63% — below the pre-registered 70% bar → corrective applied (§4.1).**
- Drop reasons: 24 citation_unverified, 13 abstained_pre_llm (both are the gates working,
  not defects: unclean teacher outputs are *excluded from training*).

---

## 2. Architecture: what the student learns vs what stays deterministic

The student replaces ONLY the generator call inside `RAGService.answer()`
(`app/services/rag_service.py`). It learns five behaviors from gate-clean teacher pairs:

1. Grounded STRICT-JSON answers over supplied retrieval context (answer, law_reference,
   action, confidence, reasoning, 6-array case analysis);
2. Citation discipline (cite only what context supports);
3. Mode-A vs Mode-B hedging (strong vs thin context);
4. Hindi answers with law refs in standard English form + English reasoning prefix;
5. Follow-up coherence (history-resolved standalone questions) — *measured 10/10 clean
   in the pilot; our strongest slice.*

**Unchanged and deterministic at serve time** (defense-in-depth stays regardless of
engine): retrieval + rerank, abstention threshold, citation verification, confidence
enforcement, current-law rails + grave-offence guard, scrubbers, classification
suppression, escalation. A weak student answer FAILS THE GATES and (in the serving
architecture, §6) falls back to Claude — trust regression risk is structurally ~zero.

---

## 3. The gate ladder (each rung cheap, falsifiable, pre-registered)

| Gate | What | Cost | Pass bar | Status |
|---|---|---|---|---|
| **Gate-0** | Eval asset: grow `data/eval/golden.jsonl` 18 → **300+** stratified rows (BNS/BNSS/BSA/CPA/labour, Hindi, abstain, IPC-era traps) + contamination blocklist wired into `export_dataset.py` | $0 | exists; ±5.7pp resolution acknowledged (−5pp detection eventually needs ~770 rows) | **NEXT** |
| **Gate-0.5** | Serving + base baselines: pull `qwen3:4b-instruct-2507-q4_K_M`, measure real laptop iGPU latency on 4k-token prompts; run BASE Qwen3-4B through `eval_student.py` (the "before" column); byte-verify Ollama TEMPLATE vs training template | $0 | measured numbers replace interpolations | pending |
| **Baseline sweep** | Haiku (teacher ceiling), base Qwen3-4B, Aalap on AIBE + 1k adalat-ai sample, closed-book AND retrieval-on | ~$10 | sets the delta to demonstrate | pending |
| **Gate-1** | 100-seed pilot: clean-keep ≥70%, cost projection | ~$1 | **DONE: 63% → corrective mix (§4.1)** | ✅ measured |
| **v0 train** | 2k clean pairs, QLoRA on Kaggle T4 (~3.5–8h, 1 session) | $0 | trains; JSON sanity on val | pending |
| **Gate-2** | Student vs Haiku on the Gate-0 harness **at q4_K_M via Ollama** (not fp16-Colab): JSON-parse ≥99%, citation-verified ≥ baseline−5pp, abstain cases safe, ZERO verdict-prediction/fabricated-precedent, Hindi format intact, rewrite-JSON 100% | $0 | all rows pass | pending |
| **Gate-3** | Measured latency decision: laptop iGPU (expect ~20–40s/answer — *llama3.2-3B measured 9.5s warm on this laptop's Radeon 740M*) | $0 | documented serve surfaces | pending |
| **v1 data+train** | 12–15k curriculum pairs (§4), 1 epoch, 1 Kaggle week | ~$65–130 sync (§7) | beats v0 on the harness | later |
| **Shadow serving** | Modal student replays recorded prod prompts, gates re-run offline; MEASURE fallback rate f | $0 (Modal $30/mo credits) | f measured, not guessed | later |
| **Canary → default** | student-first + Claude fallback on gate-fail | — | f<15%, p50 ≤30s, feedback stable | later |
| **v2 preference** | RAFT first (k=4 sample, keep gate-passing best, SFT); KTO only if a measured gap remains (gates emit unpaired verdicts — KTO's native format) | ~$5–15 | beats v1 on harness, no format drift | later |
| **v3 scale** | Qwen3-8B rung, same recipe — ONLY when a GPU serving story exists. **No continued-pretraining ever** (SaulLM needed ~30B tokens for +6%; RAG carries our knowledge) | ~2× v1 | — | later |

---

## 4. Data ladder (licenses corrected by adversarial review)

**Iron rule: NEVER train on any public benchmark item, regardless of license.** Legality
is not the bar — scorecard credibility is. Enforced via `source_manifest.json` +
contamination blocklist (SHA-256 of NFC-normalized/lowercased/punct-stripped queries +
8-gram overlap + embedding screen at cosine ~0.9) checked in `export_dataset.py`.

| Source | Size / license | Role |
|---|---|---|
| Own trust-gated distillation (this pipeline) | ~2k → 15k pairs | **TRAIN** (core) |
| IndicLegalQA | 10k, CC-BY | TRAIN seeds — but capped (§4.1): it is ALSO 82% of the retrieval corpus (three-way collision inflates internal numbers) |
| Own narrative/followup/Hindi templates | unlimited, ours | TRAIN seeds (best clean-rates: 84–100%) |
| **AIBE past papers** (~1,158 MCQs) | **CC-BY-ND, gated** | **EVAL-ONLY** (was wrongly slated for training in research draft) |
| **BhashaBench-Legal** (24,365 MCQs, 7,318 Hindi) | CC-BY-4.0 but **eval benchmark, NO train split** | **EVAL-ONLY** — it is our Hindi benchmark, never Hindi training data |
| adalat-ai/indian-legal-exam-benchmark (6,218 MCQs) | MIT | EVAL-ONLY (the public scorecard) |
| IL-TUR | CC BY-NC-SA | EVAL-ONLY (LSI task maps directly onto law_reference) |
| MILDSum (3,122 En+Hi judgment summaries) | email-gated | eval bonus (Hindi summarization) |
| NyayaAnumana / ILDC | — | **EXCLUDED entirely** (verdict prediction violates the trust contract) |

**Hindi parity:** Hindi training comes from translated own-corpus seeds + hand-authored
Devanagari templates (never shell-piped — UTF-8 files only), keeping ~25% Hindi share.

**Preference capture (build now, use later):** `generate_pairs.py` records per-sample
gate verdicts. v0 runs k=1 → only unpaired negatives (measured 37/100) — **below any DPO
floor**, so preference optimization is honestly deferred to v2 when k≥2 sampling or
shadow-traffic logs exist. RAFT-first, then KTO (unpaired-native).

### 4.1 Gate-1 corrective (applied)

Pilot measured clean-rates → new full-run mix: **indicqa 20% · narrative_en 35% ·
narrative_hi 25% · followup 15% · thin 5%** (projected clean ≈70%; also shrinks the
IndicLegalQA collision). To yield **~2,000 clean pairs**: **~2,900 seeds ≈ $19**
(measured $0.65/100 seeds, all teacher calls included) — inside the honest $12–26 band.

---

## 5. Training ladder (times corrected)

- **v0 (now):** 2k pairs × ~4.5k tok ≈ 9M tok/epoch; 2 epochs ≈ **3.5–8h on one T4** =
  one Kaggle session, $0. Unsloth QLoRA r=16, α=32, lr 2e-4, seq 8192, bs 2 × accum 4,
  `train_on_responses_only`. **Traps:** use the FIXED Qwen3-4B-Instruct-2507 chat
  template (unsloth issue #3383 — pre-fix template mis-masks assistant tokens); watch
  fp16 loss stability on T4 in the first 100 steps (Qwen3 is bf16-native); pin the
  unsloth version after one clean GGUF export; measure real tok/s in the first 30 min
  and re-plan.
- **Packing honesty:** our samples average ~4.5k tok in an 8192 window → realistic
  packing gain **1.2–1.6×**, not the advertised 3× (that's for short-sample data).
- **v1:** 12–15k pairs, 1 epoch ≈ one Kaggle week free, or $5–15 on a RunPod A40 when
  iteration speed matters. Curriculum staged by TASK (statute QA → grounded QA →
  case-analysis JSON → drafting last), 10–25% general-instruct replay against forgetting.
- **Anti-format-lock guardrails:** LoRA only (never full FT), r ≤32, ≤2 epochs, keep
  `reasoning` free-form, schema enforced at gate time not in the loss.
- **Eval compute (previously unbudgeted):** the full retrieval-on scorecard is ~9–10k
  MCQs × 12–20s ≈ **31–52 wall-clock hours per run** → per-checkpoint dev slice of
  500–1k items (~2–6h); full runs at release only.

---

## 6. Serving architecture (measured)

**Laptop (this machine, measured):** Ollama uses the Radeon 740M iGPU via Vulkan —
llama3.2-3B q4: 3,144 tok/s prefill, 63 tok/s decode, **9.5s/answer warm**. Qwen3-4B q4
interpolates to ~20–40s/answer (Gate-0.5 measures the real artifact).
**HF free Space CPU (measured proxy): 4B ≈ 5–8 min/answer → NOT a prod path, ever.**
**Live prod baseline (measured): Haiku 16.2–21.9s server-side.**

- **Prod student serving:** Modal serverless (T4 $0.000164/s; **$30/mo free credits ≈
  4,000–7,000 free queries/mo**; warm query ~$0.004–0.006) running llama.cpp
  `llama-server` + the GGUF on a Modal volume, called from the Space via a thin client
  beside `OllamaClient`. HF ZeroGPU = demo/shadow channel only.
- **Routing (the core product idea):** student-first, gate-aware fallback — generate with
  the student → run the EXISTING unchanged gates → on parse-fail / citation_verified=false
  / enforced-low-confidence, regenerate once with Claude on the identical prompt and
  re-gate. Worst case = today's quality at today's cost. **Economics (honest):** below
  ~1k queries/mo savings are <$10/mo — flip for the hedge + data flywheel, not cost;
  at 5k/mo it's ~$50→$5, at 30k/mo ~$300→$30.
- **Rollout:** 2–4 weeks SHADOW (replay recorded prompts on Modal, gates re-run offline —
  measures the real fallback rate f; any current f number is a projection, trust only
  the measurement) → 5–10% canary → default-local at f<15% and p50 ≤30s.
- **Do THIS WEEK regardless:** persist `/feedback` + (later) shadow logs to a private HF
  dataset — `app/routes/feedback.py` currently logs to ephemeral Space stdout; every lost
  day is lost flywheel data, and the flywheel (real queries → gate verdicts → preference
  data) is the only compounding asset a solo founder has.

---

## 7. Budget (corrected — no batch-price double-count)

The locked contract generates synchronously through the real `RAGService`/`ClaudeClient`
(prompt caching / Batches API don't apply to the sync path). Honest sync pricing,
**measured** at ~$6.5/1k seeds for our mix:

| Item | Cost |
|---|---|
| Gate-1 pilot (done) | **$0.65 measured** |
| v0 full run (~2,900 seeds → ~2k clean) | **~$19** |
| Baseline sweep (Haiku on AIBE + 1k adalat) | ~$10 |
| v0 training + Gate-2/3 | $0 (Kaggle) |
| v1 data (+10–13k pairs, sync) | ~$65–130 *(a Batches re-architecture would halve this for ~2–4 days of work + offline-gate revalidation — decide at v1 time, not now)* |
| v1 training | $0–15 |
| v2 RAFT sampling | ~$5–15 |
| Serving (Modal) | $0 within free credits |
| **Total to a public-scorecard v1 model** | **~$100–190 spread over 2–3 months** |

---

## 8. Public scorecard (what we publish)

Closed-book AND retrieval-on, abstentions counted as wrong, per exam:
1. AIBE past papers (~1,158 MCQs; human pass = 40%; Aalap = 25.56%; gpt-3.5 = 58.72%)
2. adalat-ai benchmark (6,218 MCQs: CLAT UG/PG, DJS, DHJS)
3. IL-TUR LSI (statute identification; published baseline mF1 28.08)
4. BhashaBench-Legal Hindi slice (Hindi/English parity, never trained on)
5. **Trust table (unique to us):** hallucinated-citation rate, abstention precision/recall,
   current-law-bridge accuracy, red-team pass rate, Hindi-format integrity — teacher vs
   base vs student columns.

Known corpus gaps that will honestly depress retrieval-on MCQ scores: no CPC, Negotiable
Instruments Act, Limitation Act, Advocates Act/BCI rules in the index yet. That is a
**corpus roadmap** (each is one `ingest_act_pdf.py` run away — see the CPA/Labour-Codes
pattern), synergistic with PAGE_INDEXING_PLAN.md (retrieval-side; independent workstream).

---

## 9. Runbook (real paths, real flags)

```bash
cd nyaysetu-backend && export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe

# 1. Seeds (deterministic, no API cost) — corrective mix lands in build_seed_queries.py
$PY scripts/distill/build_seed_queries.py --total 2900 --seed 42 --show-samples 3

# 2. Teacher pairs (REAL Haiku calls; resumable by id; hard cost stop; teacher hard-assert)
$PY scripts/distill/generate_pairs.py --seeds data/distill/seed_queries.jsonl \
    --limit 2900 --max-cost-usd 25
$PY scripts/distill/generate_pairs.py --report   # Gate-1 style readout any time

# 3. Export train/val (clean-only, 95/5 by id hash) + stats
$PY scripts/distill/export_dataset.py

# 4. Train: upload data/distill/train.jsonl + val.jsonl to Kaggle/Colab, open
#    training/nyaysetu_finetune_qwen3_4b.ipynb, Run-All → downloads
#    nyaysetu-legal-qwen3-4b-q4_k_m.gguf

# 5. Serve locally
ollama create nyaysetu-legal -f training/Modelfile   # GGUF beside the Modelfile

# 6. Gate-2: baseline once, student per checkpoint, compare
$PY scripts/distill/eval_student.py --provider claude --model claude-haiku-4-5 \
    --out data/distill/baseline_haiku.json
$PY scripts/distill/eval_student.py --provider ollama --model nyaysetu-legal \
    --out data/distill/student_v0.json
$PY scripts/distill/eval_student.py --compare data/distill/baseline_haiku.json \
    data/distill/student_v0.json
```

Ops notes: stop uvicorn before anything touching the embedded Qdrant index;
`generate_pairs.py` aborts loudly if a stray `HIGH_POWER_MODEL` would switch the teacher;
`data/distill/` is gitignored (artifacts private; also avoids false-triggering the
publish-index CI which watches `data/**`).

---

## 10. Risks (stated plainly)

1. **Anthropic usage policy on distillation** — training a model on Claude outputs may be
   restricted as "training competing AI models" under Anthropic's commercial terms. Our
   student is a domain generator for our own free product, not a general competing model,
   and the dataset stays private — but **review the current Anthropic ToS / seek written
   clarification before publicly releasing model weights**. (Publishing the scorecard is
   safe either way; the weights question is the one to clear.)
2. **Teacher drift** — Haiku updates change the target distribution; pin per-run
   (`--teacher-model` + hard assert), record model id in every pair.
3. **Small-model format-lock / forgetting** — mitigations in §5; Gate-2's rewrite-JSON
   and Hindi rows exist precisely for this.
4. **Statistical power** — until golden reaches ~770 rows, a −5pp citation regression is
   only detectable at ~±5.7pp resolution (n=300). Stated on the scorecard.
5. **Colab/Kaggle fragility** — checkpoints every 30 min to Drive/HF; resumable.
6. **BharatGen/funded competitor ships first** — mitigated by moving the scorecard
   (Gate-0 + baseline sweep) to the FRONT of the ladder, before big training spend.
7. **q4 quantization quality cliff** — Gate-2 runs against the ACTUAL q4_K_M artifact via
   Ollama, never the fp16 Colab checkpoint.
