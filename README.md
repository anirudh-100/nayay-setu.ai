# NyaySetu — Legal AI for India

A public-facing legal assistant that answers questions about Indian law in plain
language, with **verifiable citations**, runs **fully local and free**, and **abstains
rather than hallucinate** when it isn't sure.

This is a monorepo:

| Folder | What it is |
| ------ | ---------- |
| [`nyaysetu-backend/`](nyaysetu-backend/) | FastAPI **hybrid RAG** engine — InLegalBERT + Qdrant (dense) + BM25 (lexical) → RRF fusion → cross-encoder rerank → grounded, cited answers via a local Ollama LLM. Includes the IPC↔BNS current-law mapping, citation verification, and an eval harness. |
| [`nyaysetu-frontend/`](nyaysetu-frontend/) | Next.js + Tailwind chat UI — friendly, mobile-first, bilingual (English / हिन्दी), with clickable source citations. |

## Quick start

**Backend** (see [nyaysetu-backend/README.md](nyaysetu-backend/README.md)):

```bash
cd nyaysetu-backend
python -m venv .venv && .venv\Scripts\activate     # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
ollama pull mistral
python scripts/build_index.py
uvicorn app.main:app --reload                       # http://127.0.0.1:8000
```

**Frontend** (see [nyaysetu-frontend/README.md](nyaysetu-frontend/README.md)):

```bash
cd nyaysetu-frontend
cp .env.local.example .env.local
npm install
npm run dev                                          # http://localhost:3000
```

## Highlights

- **Hybrid retrieval** (dense + keyword) with cross-encoder reranking — tuned for Indian law.
- **Current-law aware** — bridges repealed IPC sections to their current BNS successors
  (date-aware, 1 July 2024 transition).
- **Trustworthy by design** — every grounded answer carries clickable citations; the engine
  verifies its own citations, downgrades unverified ones, abstains on weak retrieval, and
  points to free legal aid (NALSA 15100).
- **Extensible ingestion** — add a bare act by dropping a CSV + one `data/acts/registry.json`
  entry; no code changes.

> **Not legal advice.** This system provides legal *information* to help people understand
> their situation. The bundled BNS text and IPC↔BNS mapping are curated starters flagged for
> verification — replace with official sources before any production use.
