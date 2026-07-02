# Own Fine-Tuned Legal Model Plan (`nyaysetu-legal`)

> Status: **locked design / not started.** This is the plan doc `CLAUDE.md` §"Active
> initiatives" promises ("to be added once base-size/teacher are locked"). Base size and
> teacher are now locked (below). A future session should read this top-to-bottom, then
> confirm scope before touching files.

## 0. TL;DR — the locked decisions

| Decision | Locked value | Why |
|---|---|---|
| **Student** | **Qwen3-4B Instruct** (`Qwen/Qwen3-4B-Instruct-2507`, Apache 2.0) | The originally-noted Qwen2.5-3B is under the **Qwen Research License (non-commercial)** — unusable for a product. Qwen3-4B is Apache 2.0, non-thinking instruct (clean JSON generator), and small enough for free-tier training + CPU GGUF serving. |
| **Teacher** | **`claude-haiku-4-5`** via the existing `app/services/llm_service.ClaudeClient` | It is the **audited production engine** (the live HF-Spaces deployment answers with Haiku). Distilling from the engine we already trust means the dataset inherits the audited behavior. $1.00/MTok in, $5.00/MTok out. |
| **Dataset** | ~**2,400 seeds → ~2,000 clean pairs** | Target ~$15 API budget (see §6 — honest range is $12–24; the 100-seed pilot measures the real per-pair cost before the full run). |
| **Training** | **QLoRA via Unsloth** on free **Colab T4** (fallback: Kaggle 2×T4) | Zero training cost. 4B in 4-bit fits a T4 with 8K sequence length. |
| **Export / serving** | **GGUF `q4_K_M`** → file `nyaysetu-legal-qwen3-4b-q4_k_m.gguf` → **Ollama model `nyaysetu-legal`** | Drops into the existing `OllamaClient` path with a `.env` change only (`LLM_PROVIDER=ollama`, `OLLAMA_MODEL=nyaysetu-legal`). No code change to serve. |
| **Production** | **Stays on Haiku** until Gate-2 passes **and** a serving story exists (Gate-3) | Be honest: 4B q4 on a laptop CPU ≈ 8–15 tok/s → ~45–90 s per answer; HF Space free CPU (2 vCPU) is likely 2–6 tok/s → minutes per answer, too slow for prod. |

```
knowledge  = retrieval corpus (models/index, 12,201 chunks)   → UNCHANGED by this plan
behavior   = the generator (JSON answer over given context)   → what the student learns
trust      = deterministic gates in rag_service.py            → UNCHANGED, run over ANY engine
```

---

## 1. Goal & positioning

NyaySetu is *a legal retrieval engine with a bot front-end*. Its legal **knowledge** lives
in the retrieval corpus (`models/index`: IndicLegalQA + 13 acts + judgments + guides), not
in model weights — and this plan keeps it that way. The student model does **not** learn
Indian law. It learns the **generator behavior** that today only Claude Haiku performs
reliably and `mistral`/`llama3.2:3b` perform poorly:

1. **Grounded JSON answers over given context** — Mode A: use ONLY the retrieved
   `[LABEL]`-tagged chunks, quote punishments verbatim, lead with current law
   (BNS/BNSS/BSA over IPC/CrPC/IEA).
2. **Citation discipline** — `law_reference` copied verbatim from a context `[LABEL]`
   (never an invented section; never BNSS shortened to "BNS").
3. **Mode-A / Mode-B hedging** — when context is thin/off-topic, answer with
   `"Typically under Indian law,"` + a broad reference + reasoning starting
   `"No strong context, used general principles"`, without inventing section numbers.
4. **Hindi output** — Devanagari `answer`/`action`/analysis prose with law references
   kept in the standard English form (`"BNS Section 103"`), per `_HINDI_INSTRUCTION`.
5. **Case-analysis structure** — the six `analysis` arrays with impersonal stems
   ("The law allows…", "A court may…"), no offence-classification labels, no outcome
   prediction, no invented precedents.

**What the student deliberately does NOT learn:**

- **Abstention on junk.** Abstention stays a **pre-LLM retrieval gate** at serve time
  (`min_rerank_score=0.0` in `app/config/settings.py`; `RAGService._abstain()` returns
  before any LLM call). The student never sees junk queries, so it doesn't need to learn
  junk-refusal. Thin-context **Mode-B hedging IS learned** (that path does reach the LLM).
- **Legal knowledge recall.** Corpus growth (new acts, page indexing) never requires
  retraining.
- **Current-law mapping.** `app/rag/law_map.py` + `_prefer_current_reference` are
  deterministic code, engine-independent.

**Why build it at all:**

- **The local/free engine.** It replaces `mistral` / `llama3.2:3b` as the
  `LLM_PROVIDER=ollama` default — today those produce weak citations and flaky JSON;
  a distilled 4B should behave like a slow Haiku *on this one task*.
- **A hedge against API dependency.** If the Anthropic key, budget, or rate limit dies,
  the product degrades to "slower" instead of "dead".
- **Zero serving-code change.** `OllamaClient.generate_json()` (`/api/generate`,
  `format=json`, `temperature 0.2`) already exists; the student slots in by name.

**Where it serves (honest):**

| Surface | Verdict |
|---|---|
| Laptop dev / offline demo (Ollama, CPU) | ✅ primary home. ~45–90 s/answer at 8–15 tok/s for the ~500–900-token JSON. Fine for dev + demos, not delightful. |
| HF Space free CPU (the live backend host) | ❌ likely 2–6 tok/s on 2 vCPU → 2–5 min/answer. Not a prod path. |
| Production | Stays **`claude-haiku-4-5`** until Gate-2 passes AND a real serving story exists (small GPU instance, or a user-visible "free slow mode" toggle). This plan does not pretend otherwise. |

All trust gates in `RAGService.answer()` (citation verification, deterministic confidence,
reporter-citation scrub, grave-offence guard, analysis master gate) run **after** the LLM
regardless of engine — the student is wrapped by the same defence-in-depth as Haiku.

---

## 2. Distillation design

### 2.1 Pairs come from the REAL pipeline, never a reimplementation

Each training pair is produced by running a seed query through the actual
`RAGService.answer(query, language, history)` with the actual `ClaudeClient` — real
hybrid retrieval, real current-law expansion, real BNSS procedure injection, real prompt
(`PROMPT_TEMPLATE` + optional PROCEDURE CONTEXT + optional `_HINDI_INSTRUCTION`), and the
real trust gates grading the result. The generator script wraps the LLM in a thin
recorder:

```python
class RecordingLLM:
    """Wraps ClaudeClient; records every (prompt, parsed_json) generate_json call."""
    def __init__(self, inner):
        self.inner, self.calls = inner, []
    def generate_json(self, prompt: str) -> dict:
        out = self.inner.generate_json(prompt)
        self.calls.append((prompt, out))
        return out

svc = RAGService(llm=RecordingLLM(ClaudeClient()))
resp = svc.answer(seed["query"], seed["language"], history)   # history = [ConversationTurn(**t) ...]
```

- The **MAIN generation call** is identified by prefix: the one call whose prompt starts
  with `"You are an AI legal assistant for Indian law."` (the first line of
  `PROMPT_TEMPLATE` in `app/services/rag_service.py`). `answer()` makes at most one such
  call per query. Its exact prompt string and raw parsed JSON become
  `pairs.jsonl.prompt` / `pairs.jsonl.target_json`.
- Rewrite calls (`_standalone_query` / `_recovery_query`, prompts starting `"Rewrite…"` /
  `"You convert a person's LEGAL PROBLEM…"`) are **not** training targets, but their
  tokens **are** counted into `est_tokens` (true cost accounting).
- `final.*` is taken from the returned `AskResponse` — i.e. **post-gate, post-scrub**
  values (`answer`, `law_reference`, `confidence`, `citation_verified`, `abstained`).

### 2.2 The contract rule: train ONLY on gate-clean examples

```
clean = (not abstained)
        AND citation_verified
        AND final.answer == target_json.get("answer")     # scrubbers changed nothing
        AND final.confidence in ("high", "medium")
```

Rationale per clause:

| Clause | What it filters out |
|---|---|
| `not abstained` | Retrieval-gate abstentions (no LLM call happened; `prompt=""`, `target_json={}`, `drop_reason="abstained"`). |
| `citation_verified` | The hallucination signal — teacher cited a section not in the retrieved sources. Never teach that. |
| `final.answer == target_json["answer"]` | If `_scrub_reporter_citations` (or any fallback) altered the served answer, the raw teacher output contains a fabrication-adjacent token — drop it (`drop_reason="scrubbed_answer_mismatch"`) rather than train on pre-scrub text. |
| `confidence in ("high","medium")` | `high` = Mode A verified; `medium` = clean Mode-B hedge (this is how hedging is learned). `low` = downgraded/uncertain — drop (`drop_reason="low_confidence"`). |

`LLMError` / transport failures are recorded as `clean=false, drop_reason="llm_error"`
so the run stays resumable and the cost stays visible.

### 2.3 Serve-time division of labor (unchanged)

```
query → [pre-LLM] rewrite → hybrid retrieve → abstain-if-weak   ← stays deterministic + gate
      → [LLM]     PROMPT_TEMPLATE over context → JSON            ← the ONLY thing distilled
      → [post-LLM] confidence enforcement, citation verify,
                   scrubbers, current-law correction, analysis gate ← stays deterministic
```

**Known risk (tracked in Gate-2):** with `LLM_PROVIDER=ollama`, `RAGService` uses the
same client for the small **query-rewrite** calls (`{"q": "..."}` JSON). Those are not in
the training set. Base Qwen3-4B-Instruct handles them fine untuned, and LoRA on the
generator task should not destroy that — but Gate-2 explicitly checks that rewrite calls
still return parseable `{"q": ...}` after fine-tuning.

---

## 3. Dataset mix (~2,400 seeds)

| kind | share | count | language | source / construction |
|---|---|---|---|---|
| `indicqa` | 40% | ~960 | en | Questions sampled from `data/indiclegalqa/IndicLegalQA Dataset_10K.json` (fields `question`/`case_name`). Stratified: max 1 question per `case_name`, dedup by normalized text. These retrieve their own QA chunks strongly → teach Mode-A grounding + judgment-vs-statute discipline. |
| `narrative_en` | 25% | ~600 | en | Lay first-person situations ("the shopkeeper sold me a defective mixer and won't refund…") built from a curated template bank × slot fill across domains: offences (theft/cheating/hurt/threat/dowry/stalking), consumer (CPA 2019), labour (Code on Wages, IR Code), cheque bounce, tenancy, family, cyber. Exercises the narrative rewrite + case-analysis path. |
| `narrative_hi` | 20% | ~480 | hi | Devanagari versions of the same situation space (independent phrasings, not translations of the en set). **Hindi gotcha:** these are authored/generated inside Python and written straight to the UTF-8 JSONL — never passed through shell echo/vars. |
| `followup` | 10% | ~240 | en/hi | Two-turn seeds: `history` = one realistic user+assistant turn, `query` = a dependent follow-up ("what's the punishment for that?", "can I get bail?"). Exercises `_standalone_query` resolution; the main prompt then carries the resolved question (`prompt_question = search_query` when history is present). |
| `thin` | 5% | ~120 | en | Legal-but-weakly-covered queries that clear the abstain bar yet have off-point context (niche acts, vague rights questions) → teacher answers in Mode B. This is where **medium-confidence hedging** is learned. |

`domain` tags each seed (e.g. `criminal`, `consumer`, `labour`, `family`, `property`,
`cyber`, `procedure`, `constitutional`, `judgment_qa`) for stratified stats and spot-reads.

**Held-out — never in seeds:**

- `data/eval/golden.jsonl` (18 rows) — Gate-2 eval set. Seed construction runs an exact
  + normalized string-match filter against every golden `query`; overlapping *topics*
  are allowed (theft questions exist in both worlds), identical queries are not.
- `scripts/answer_eval.py` `GOLD` (13 rows incl. 2 abstain probes) — same filter.

**Seed ID (pinned recipe, all components MUST use it):**

```python
sha1_12 = hashlib.sha1(
    f"{query}|{language}|{json.dumps(history, ensure_ascii=False, sort_keys=True)}"
    .encode("utf-8")
).hexdigest()[:12]
```

---

## 4. Pipeline stages + file/script layout

New files this initiative owns (nothing existing is modified):

```
data/distill/
  seed_queries.jsonl        # stage 1 output  (interface #1)
  pairs.jsonl               # stage 2 output  (interface #2, append-only, resumable by id)
  train.jsonl  val.jsonl    # stage 3 output  (interface #3, Unsloth chat format)
  dataset_stats.md          # stage 3 output  (counts, drop reasons, est cost)
data/eval/
  redteam.jsonl             # ~20 verdict/precedent/Hindi probes for Gate-2
scripts/
  distill_make_seeds.py     # stage 1: build + validate seed_queries.jsonl (golden filter, mix, ids)
  distill_generate_pairs.py # stage 2: seeds -> RAGService(RecordingLLM(ClaudeClient())) -> pairs.jsonl
                            #   flags: --limit N (pilot), --only-kind K; skips ids already in pairs.jsonl
  distill_build_dataset.py  # stage 3: pairs.jsonl -> train/val (clean only, 95/5 by id-hash) + stats
                            #   flag: --stats-only (Gate-1 readout without writing train/val)
  distill_eval_student.py   # stage 5: golden.jsonl + redteam.jsonl over /ask; compares two
                            #   captured runs (student vs Haiku baseline); prints Gate-2 verdict
notebooks/
  train_nyaysetu_qwen3_4b.ipynb   # stage 4: Colab/Kaggle QLoRA + GGUF export
models/ollama/
  Modelfile                 # FROM nyaysetu-legal-qwen3-4b-q4_k_m.gguf (+ ChatML template, num_ctx)
```

### Shared interface contract (normative — restated exactly)

1. **`data/distill/seed_queries.jsonl`** — one JSON per line:
   `{"id": "<sha1-12 of query|language|history>", "query": str, "language": "en"|"hi",
   "history": [{"role":"user"|"assistant","content":str}, ...]` (empty list if
   single-turn), `"domain": str, "kind":
   "indicqa"|"narrative_en"|"narrative_hi"|"followup"|"thin"}`
2. **`data/distill/pairs.jsonl`** — one JSON per line (append-only, resumable by id):
   `{"id"` (same as seed), `"domain", "kind", "language", "query", "history",
   "prompt": str` (the EXACT final prompt string sent to the teacher for the MAIN
   generation call), `"target_json": dict` (the teacher's RAW parsed JSON reply for that
   call), `"final": {"answer": str, "law_reference": str, "confidence": str,
   "citation_verified": bool, "abstained": bool}, "clean": bool,
   "drop_reason": str|null, "est_tokens": {"in": int, "out": int}}` — token estimate =
   `ceil(chars/4)`, summed over **all** teacher calls for the seed (rewrites included).
   Clean rule exactly as §2.2. Abstained seeds: `prompt=""`, `target_json={}`.
3. **`data/distill/train.jsonl` + `val.jsonl`** — Unsloth chat format, one per line:
   `{"messages": [{"role":"user","content": <prompt>},
   {"role":"assistant","content": <json.dumps(target_json, ensure_ascii=False)>}]}` —
   built ONLY from clean pairs; deterministic 95/5 split by id hash
   (`int(id, 16) % 20 == 0` → val); plus `dataset_stats.md` (counts by
   kind/language/confidence, drop reasons, est cost). No system message: at serve time
   `OllamaClient` sends none, and `format=json` constrains the output shape.
4. **Student model:** Qwen3-4B (Apache 2.0), Unsloth 4-bit, GGUF `q4_K_M` output named
   `nyaysetu-legal-qwen3-4b-q4_k_m.gguf`, served via Ollama model name **`nyaysetu-legal`**.
5. **Teacher:** `claude-haiku-4-5` through the EXISTING pipeline
   (`RAGService` + `ClaudeClient`) — never a reimplementation of retrieval or prompting.

### Stage notes (grounded gotchas)

- **Teacher pinning is mandatory.** `app/config/settings.py` defaults
  `high_power_model="claude-opus-4-8"` and the local `.env` does **not** set
  `HIGH_POWER_MODEL`. Every pair-generation run MUST `export HIGH_POWER_MODEL=claude-haiku-4-5`
  (pydantic-settings reads env over defaults) — otherwise pairs silently come from Opus at
  5× the price and off-contract. `distill_generate_pairs.py` must hard-assert
  `settings.high_power_model == "claude-haiku-4-5"` before the first call. Optionally
  `export ANTHROPIC_THINKING=off` to skip the one auto-disable retry Haiku triggers.
- **Qdrant file lock.** `RAGService` opens the embedded Qdrant under `models/index/` —
  the same lock `uvicorn` holds. Stop the API before running `distill_generate_pairs.py`.
  Do NOT rebuild the index; it exists (12,201 chunks incl. CPA + Labour Codes).
- **History typing.** `_standalone_query` reads `t.role` / `t.content` attributes —
  convert seed history dicts to `app/schemas/ask.ConversationTurn` before calling
  `answer()`.
- **Training config (stage 4):** base `Qwen/Qwen3-4B-Instruct-2507` via Unsloth
  (`unsloth/Qwen3-4B-Instruct-2507`, `load_in_4bit=True`; the `-unsloth-bnb-4bit`
  prequant also works). `max_seq_length=8192` (real prompts run ~3.5–8K tokens with
  context + procedure arc), LoRA `r=16, alpha=16`, lr `2e-4`, 2 epochs, per-device batch
  1 × grad-accum 8 (~475 steps on ~1,900 train rows), `train_on_responses_only` so loss
  is on the assistant JSON. Expect roughly 3–6 h on a free T4; Kaggle (2×T4, 30 h/wk) is
  the fallback if Colab disconnects. Export:
  `model.save_pretrained_gguf(..., quantization_method="q4_k_m")` (~2.5 GB) and push to a
  private HF repo for download (more reliable than browser download from Colab).
- **`models/ollama/Modelfile`:** `FROM ./nyaysetu-legal-qwen3-4b-q4_k_m.gguf`, explicit
  ChatML (Qwen3) `TEMPLATE`, and **`PARAMETER num_ctx 8192`** — Ollama's default context
  is far smaller and would silently truncate the retrieval context (the classic failure
  mode here). Temperature is already sent per-request by `OllamaClient` (`0.2`).

---

## 5. GATES — nothing ships past a failed gate

### Gate-1 — Pilot (before spending the full budget)

Run: `distill_generate_pairs.py --limit 100` on a mix-proportional 100-seed sample.

- **Pass:** clean-keep rate **≥ 70%** (70+ of 100 pairs `clean=true`), AND a **manual
  spot-read of 10 pairs** (2 per kind) finds: answer grounded in the recorded prompt's
  context, `law_reference` matches a `[LABEL]`, Hindi pairs are natural Devanagari with
  standard-form law refs, analysis arrays respect the impersonal stems.
- **Also measured:** real per-pair cost from `est_tokens` → recompute the full-run
  projection (§6). If projected full run > **$25**, shrink the seed count to fit rather
  than bust the budget.
- **Fail →** fix seeds/prompts (usually: thin seeds abstaining instead of Mode-B, or
  IndicLegalQA sampling pulling unanswerable trivia) and re-pilot. Do not run 2,400 seeds
  on a failing recipe.

### Gate-2 — Student vs Haiku baseline (before any default flip)

Capture two full runs over the served API (`scripts/answer_eval.py` +
`distill_eval_student.py` over `data/eval/golden.jsonl` and `data/eval/redteam.jsonl`):
once with `.env` → Haiku (baseline), once with `LLM_PROVIDER=ollama`,
`OLLAMA_MODEL=nyaysetu-legal`.

| Check | Pass bar |
|---|---|
| JSON-parse rate (no `LLMError: Could not parse LLM JSON` across all calls) | **≥ 99%** |
| `citation_verified` rate on substantive answers | **≥ baseline − 5 pp** |
| Verdict-prediction probes ("will I win?", "how many years will he get?") | **zero** outcome predictions surviving in `answer`/`analysis` |
| Fabricated-precedent probes ("which SC judgment supports me?") | **zero** invented case names / reporter citations in served output |
| Hindi probes | Devanagari answer, `law_reference` still standard English form (`"BNS Section 103"`), `reasoning` still starts `Used …` / `No strong context …` |
| Golden section accuracy (expected section in `law_reference`+`answer`) | ≥ baseline − 10 pp (quality, not trust — report, don't hard-fail) |
| Rewrite calls (`{"q": ...}`) still parse under the fine-tuned model | 100% parseable |
| Abstain probes (non-legal queries) | still abstain — this is the pre-LLM gate, so a failure here means a harness bug, not a model bug |

Trust rows are hard gates; quality rows are reported deltas. Note the safety asymmetry:
even where the student is worse, unverified citations become `confidence=low` + legal-aid
escalation, and the analysis block suppresses itself — the gates fail safe.

### Gate-3 — Measured latency + serving decision

Measure on the real laptop via `/ask` (`response_time_ms` is already in `AskResponse`),
10 queries warm, en + hi:

| Measured median | Decision |
|---|---|
| ≤ 60 s | `nyaysetu-legal` becomes the default **local dev/demo** engine (`.env.example` note); prod unchanged. |
| 60–120 s | Local engine for offline demo only; dev default stays `llama3.2:3b` for iteration speed. |
| > 120 s | Park serving; keep the GGUF as the API-dependency hedge artifact. |
| Prod flip (any case) | Requires Gate-2 pass **and** a paid/GPU serving story with measured < 15 s median — explicitly out of scope for this plan. |

---

## 6. Cost table (claude-haiku-4-5: $1.00/MTok in, $5.00/MTok out)

Estimator = the contract's `ceil(chars/4)`. Per-pair assumptions: main prompt =
`PROMPT_TEMPLATE` (measured: 6,632 chars ≈ 1.66K est-tok) + 6 context chunks (~1.2–3K tok) + procedure arc on criminal
seeds (+~1.2K) + Hindi suffix on hi seeds (+~0.25K); output JSON ~0.5–0.9K tok; rewrite
calls ~0.5K/40 tok on the ~60% of seeds that trigger them.

| Item | Calls | Est. tokens | Cost |
|---|---|---|---|
| Gate-1 pilot (100 seeds) | ~160 | ~0.55M in / 0.07M out | **~$0.90** |
| Full run (2,400 seeds) — **low** (short contexts dominate) | ~3,800 | ~9.5M in / 1.3M out | ~$16 |
| Full run — **expected** | ~3,800 | ~11.5M in / 1.6M out | **~$15–20** |
| Full run — **high** (long statute chunks + procedure arcs) | ~3,800 | ~15.5M in / 1.9M out | ~$25 |
| Gate-2 baseline + red-team captures (2 × ~38 queries; only the Haiku run bills) | ~90 | ~0.35M in / 0.04M out | ~$0.55 |
| Training (Colab T4 / Kaggle) | — | — | $0 |
| **Total** | | | **≈ $15 target, honest range $12–26** — Gate-1 measures the real number and the >$25 projection rule (§5) trims seeds to fit |

Notes: no prompt-caching win is available (the volatile context sits near the top of
`PROMPT_TEMPLATE`, so there is no stable ≥1K-token prefix), and the Batches API is out of
bounds (contract: teacher calls go through the EXISTING `ClaudeClient`, synchronously,
so every gate sees exactly production behavior). Resumability (`pairs.jsonl` append-only,
skip existing ids) means a crashed run never re-bills completed seeds.

---

## 7. Relationship to `PAGE_INDEXING_PLAN.md`

**Independent workstreams; different halves of the engine.**

| | Page indexing plan | This plan |
|---|---|---|
| Changes | **Retrieval**: dense InLegalBERT+Qdrant → OpenSearch inverted index | **Generation**: which LLM writes the JSON |
| Keeps | rerank, RRF, citation verification, abstention, `law_map`, prompt | retrieval, all trust gates, prompt |
| Shared seam | `HybridRetriever.retrieve()` → `list[RetrievedChunk]` → `PROMPT_TEMPLATE` — both plans leave this seam's shape intact |

Sequencing: either can land first. Training pairs generated against today's retrieval
remain valid after the OpenSearch migration because the student learns *grounded JSON over
whatever context it is given*, not the retrieval itself. One caution: if page indexing
ever changes chunk **label formats** (`reference_label()`) or typical chunk length, re-run
Gate-2 on the migrated stack; if labels changed shape, top up with a small regenerated
batch. Conversely, this plan must not tune anything retrieval-side, so the page-indexing
work never invalidates the student.

---

## 8. Runbook — zero → dataset → Colab → GGUF → `ollama create` → eval

Git Bash on Windows; repo root `c:/Users/aniru/Downloads/convoia-ai/nyaysetu-backend`.
**Stop `uvicorn` before stages 1–3 (embedded Qdrant file lock).**

```bash
cd "c:/Users/aniru/Downloads/convoia-ai/nyaysetu-backend"
export PYTHONIOENCODING=utf-8

# ── Stage 0: sanity (no API cost) ────────────────────────────────────────────
.venv/Scripts/python.exe -c "from app.services.llm_service import get_llm; print(type(get_llm()).__name__)"
#   expect: ClaudeClient   (LLM_PROVIDER=claude in .env)
ls models/index/bm25.pkl models/index/qdrant        # index exists — do NOT rebuild

# ── Stage 1: seeds ───────────────────────────────────────────────────────────
.venv/Scripts/python.exe scripts/distill_make_seeds.py
#   -> data/distill/seed_queries.jsonl (~2,400 rows; golden.jsonl exact-match filtered;
#      Hindi rows written as UTF-8 by Python — never via shell vars)

# ── Stage 2a: Gate-1 pilot (~$1 of real Haiku calls) ────────────────────────
export HIGH_POWER_MODEL=claude-haiku-4-5     # REQUIRED: settings default is Opus
export ANTHROPIC_THINKING=off                # optional: skip the Haiku thinking-retry
.venv/Scripts/python.exe scripts/distill_generate_pairs.py --limit 100
.venv/Scripts/python.exe scripts/distill_build_dataset.py --stats-only
#   GATE-1: clean rate >= 70%? spot-read 10 pairs. Projected full cost <= $25?

# ── Stage 2b: full run (resumable; re-running skips finished ids) ────────────
.venv/Scripts/python.exe scripts/distill_generate_pairs.py

# ── Stage 3: dataset ─────────────────────────────────────────────────────────
.venv/Scripts/python.exe scripts/distill_build_dataset.py
#   -> data/distill/train.jsonl  val.jsonl  dataset_stats.md   (~2,000 clean pairs, 95/5)

# ── Stage 4: train on Colab T4 (free) ────────────────────────────────────────
#   Upload train.jsonl + val.jsonl; run notebooks/train_nyaysetu_qwen3_4b.ipynb:
#     unsloth FastLanguageModel "unsloth/Qwen3-4B-Instruct-2507", load_in_4bit,
#     max_seq_length=8192, LoRA r=16, 2 epochs, train_on_responses_only
#     -> model.save_pretrained_gguf(..., quantization_method="q4_k_m")
#     -> push nyaysetu-legal-qwen3-4b-q4_k_m.gguf (~2.5 GB) to a private HF repo

# ── Stage 5: local model ─────────────────────────────────────────────────────
#   Download the GGUF next to models/ollama/Modelfile, then:
ollama create nyaysetu-legal -f models/ollama/Modelfile
ollama run nyaysetu-legal    # smoke: paste a captured pairs.jsonl prompt, expect JSON

# ── Stage 6: Gate-2 + Gate-3 eval ────────────────────────────────────────────
# 6a. Haiku baseline capture (API running with .env as-is: LLM_PROVIDER=claude):
.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000 &
.venv/Scripts/python.exe scripts/answer_eval.py
.venv/Scripts/python.exe scripts/capture_answers.py out/haiku_answers.json haiku
.venv/Scripts/python.exe scripts/distill_eval_student.py --capture out/haiku_eval.json
# 6b. Student run: stop uvicorn; set in .env: LLM_PROVIDER=ollama, OLLAMA_MODEL=nyaysetu-legal;
#     restart uvicorn, then:
.venv/Scripts/python.exe scripts/answer_eval.py
.venv/Scripts/python.exe scripts/capture_answers.py out/student_answers.json nyaysetu-legal
.venv/Scripts/python.exe scripts/distill_eval_student.py --capture out/student_eval.json \
    --baseline out/haiku_eval.json
#   GATE-2 verdict printed (trust hard-fails vs quality deltas)
#   GATE-3: median response_time_ms from the student capture -> §5 decision table
```

Hindi verification at every stage follows the project gotcha: assert Devanagari by
reading the UTF-8 files from Python (`_DEVANAGARI`-style regex), never by echoing
through the shell.

---

## 9. Risks — stated plainly

1. **Latency is the product-killer, not quality.** Even a Gate-2-passing student at
   45–90 s/answer only earns the local/hedge role. Budget no expectations beyond that
   until there is GPU serving.
2. **Budget variance.** The honest full-run range is $12–26 against a ~$15 target; the
   pilot's measured `est_tokens` + the trim rule is the control, not optimism.
3. **Rewrite-call regression.** Fine-tuning could dent the untuned `{"q": ...}` rewrite
   behavior the serve path also routes through the student — explicitly gated in Gate-2.
4. **Teacher drift.** If production moves off `claude-haiku-4-5`, pairs remain valid (the
   contract is behavioral, gate-checked), but re-baseline Gate-2 against the new prod
   engine before comparing.
5. **Context truncation at serve time.** A missing `num_ctx 8192` in the Modelfile
   silently amputates the retrieved context and produces confident ungrounded answers —
   the gates would catch it as a citation-verification collapse, but check the Modelfile
   first when Gate-2 tanks.
6. **License hygiene.** Qwen3-4B is Apache 2.0 (ship-safe). Do not swap in Qwen2.5-3B
   (research/non-commercial) or any research-licensed checkpoint without re-clearing this
   table.
