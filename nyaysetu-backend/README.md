# NyaySetu Backend

> *Aapka Kanoon, Aapke Haath* — an AI legal assistant for India, built for ordinary citizens.

A FastAPI backend with a **hybrid Retrieval-Augmented Generation (RAG)** pipeline that
answers questions about Indian law with **verifiable citations**, runs **fully local
and free**, and **abstains rather than hallucinate** when it isn't sure.

## What makes the engine strong

| Stage | What it does | Why |
| ----- | ------------ | --- |
| **Dense retrieval** | InLegalBERT embeddings in a persisted **Qdrant** store, metadata-filtered | Understands Indian legal vocabulary; filterable by act/jurisdiction/language |
| **Lexical retrieval** | **BM25** over the same corpus | Nails exact tokens dense search misses — section numbers ("420"), citations, dates |
| **Fusion** | **Reciprocal Rank Fusion (RRF)** of both rankings | Scale-free way to combine semantic + keyword signals |
| **Rerank** | Cross-encoder re-scores the shortlist (≈50 → top 6) | The LLM only sees the best handful — precision here drives answer quality |
| **Grounded generation** | Local LLM via Ollama, strict prompt, **citations attached** | Answers quote sections verbatim from retrieved context |
| **Abstention** | If retrieval is weak, skip the LLM and point to free legal aid | A wrong fake section is worse than an honest "I'm not sure" |

All processing is local: queries, embeddings, reranking, and LLM inference never leave
the machine. No paid APIs.

## Project structure

```
nyaysetu-backend/
├── app/
│   ├── main.py                 # FastAPI app; loads persisted indexes at startup
│   ├── config/settings.py      # all engine knobs (env-driven)
│   ├── routes/ask.py           # POST /ask
│   ├── schemas/ask.py          # request/response models (+ citations, escalation)
│   ├── services/
│   │   ├── llm_service.py      # Ollama HTTP client (format=json)
│   │   └── rag_service.py      # orchestrates retrieve → prompt → cite → (abstain)
│   └── rag/                     # the engine
│       ├── models.py           # Chunk / Citation / RetrievedChunk (metadata contract)
│       ├── embedder.py         # InLegalBERT sentence embeddings (singleton)
│       ├── vector_store.py     # persisted Qdrant (embedded/local mode)
│       ├── lexical_store.py    # BM25 sparse index
│       ├── reranker.py         # cross-encoder reranker
│       ├── retriever.py        # hybrid: dense + BM25 → RRF → rerank
│       ├── loaders.py          # raw files → tagged chunks (IPC/QA/guides)
│       └── pipeline.py         # load → embed → persist
├── data/
│   ├── ipc/                    # IPC sections CSV (marked repealed; BNS coming in Phase 2)
│   ├── indiclegalqa/           # IndicLegalQA dataset (court Q&A)
│   └── corpus/                 # plain-language legal guides (.md/.txt)
├── models/index/               # persisted Qdrant collection + bm25.pkl
├── scripts/build_index.py      # (re)build the indexes
├── requirements.txt
└── .env.example
```

## Setup

### 1. Python dependencies

```bash
cd nyaysetu-backend
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

### 2. Install and run Ollama

Download from https://ollama.com, then pull the model:

```bash
ollama pull mistral
```

Ollama serves on `http://127.0.0.1:11434` by default.

### 3. (Optional) Configure environment

```bash
cp .env.example .env            # edit to override any default
```

### 4. Build the indexes

```bash
python scripts/build_index.py
```

This loads `data/ipc`, `data/indiclegalqa`, and `data/corpus`, tags every chunk with
its legal metadata, embeds with **InLegalBERT**, and writes the **Qdrant** + **BM25**
indexes to `models/index/`. The first run downloads InLegalBERT + the reranker from
Hugging Face (cached afterwards). Embedding ~10k chunks on CPU takes a few minutes.

> The API and the build script both open the embedded Qdrant file — **build first,
> then start the server** (don't run them at the same time).

### 5. Run the server

```bash
uvicorn app.main:app --reload
```

API: http://127.0.0.1:8000 — Swagger UI: http://127.0.0.1:8000/docs

## Usage

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the punishment for cheating under IPC 420?"}'
```

Response:

```json
{
  "answer": "...",
  "law_reference": "IPC Section 420",
  "action": "...",
  "confidence": "high",
  "reasoning": "Used IPC Section 420 from context",
  "citations": [
    {
      "label": "IPC Section 420",
      "source_type": "statute",
      "snippet": "IPC Section 420: Cheating and dishonestly inducing delivery of property...",
      "url": "https://devgan.in/ipc/section/420/",
      "code_status": "repealed"
    }
  ],
  "abstained": false,
  "escalation": null,
  "disclaimer": "This is for informational purposes only and not legal advice. ...",
  "response_time_ms": 1842
}
```

## Adding to the knowledge base

Drop `.txt`/`.md` files into `data/corpus/` (plain-language summaries with explicit
section references work best) and re-run `python scripts/build_index.py`.

## Tuning

Set any of these in `.env` (see `.env.example` for the full list):

| Setting            | Default | Effect |
| ------------------ | ------- | ------ |
| `EMBEDDING_MODEL`  | `law-ai/InLegalBERT` | Dense encoder. Swap to an `e5` model + prefixes for multilingual. |
| `RERANK_MODEL`     | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Precision reranker. |
| `USE_RERANKER`     | `true`  | Disable to trade quality for speed. |
| `FETCH_K`          | 30      | Candidates pulled from each retriever before fusion. |
| `TOP_K`            | 6       | Chunks finally shown to the LLM. |
| `MIN_RERANK_SCORE` | -10.0   | Below this top score the engine abstains. Raise for stricter abstention. |
| `OLLAMA_MODEL`     | mistral | Any Ollama model; heavier models give better answers. |

## Safety & trust

- **Citations on every grounded answer** — section/case + click-through link.
- **Citation verification** — if the model cites a section that wasn't in the retrieved
  sources (a hallucination signal), confidence is downgraded and the answer escalates.
- **Current-law first** — when a query hits a repealed IPC section, the engine pulls in
  the current **BNS** successor and leads with it; `current_law_note` bridges old→new
  with the 1 July 2024 transition rule.
- **Abstention** — weak retrieval skips the LLM entirely and points to free legal aid
  (DLSA / NALSA helpline **15100**) instead of guessing.
- **Deterministic confidence** — derived from how the answer was grounded, not the
  LLM's self-rating.
- **Not legal advice.** Every response includes a disclaimer.

## Current-law mapping (IPC ↔ BNS)

The Bharatiya Nyaya Sanhita (BNS) replaced the IPC for offences on/after **1 July 2024**.
The engine carries a curated `data/mappings/ipc_bns.json` table and a `data/bns/` corpus
so it speaks current law while still answering historic (IPC) matters.

> ⚠️ The bundled BNS text and mapping are a **curated starter, flagged `verified: false`**.
> Replace `data/bns/bns_sections.csv` with the official Bharatiya Nyaya Sanhita, 2023 bare
> act and verify `data/mappings/ipc_bns.json` before production use, then re-run the build.

## Evaluation

Measure retrieval quality (Hit@k + MRR) against a golden set — catches regressions when
you change embeddings, the reranker, fusion, or the corpus:

```bash
python scripts/eval.py            # retrieval metrics (needs the built index, not Ollama)
python scripts/eval.py --e2e      # also runs the full LLM answer (needs Ollama)
```

Verify that citation links still resolve (run after changing the corpus or link logic):

```bash
python scripts/check_links.py     # checks every BNS deep-link + a sample of search links
```

## Roadmap

- **Phase 1 (done):** hybrid retrieval + rerank + persisted indexes + InLegalBERT +
  verifiable citations + abstention.
- **Phase 2 (done):** current-law layer — BNS corpus + date-aware IPC↔BNS mapping +
  cross-reference expansion + citation verification + eval harness.
- **Phase 3:** Next.js frontend — friendly, multilingual, mobile-first UI.
- **Later:** BNSS/BSA + judgments ingestion; multilingual + voice (Bhashini).
