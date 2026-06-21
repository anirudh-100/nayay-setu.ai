# Deploying NyaySetu (web app)

This takes the web app live: **Next.js frontend on Vercel** + **FastAPI backend on
Hugging Face Spaces** (free CPU, 16 GB RAM) + **Claude Opus 4.8** as the answering
engine. Infra cost is **$0** — you pay only for Claude usage.

```
  Browser ── https ──▶  Vercel (Next.js)  ── https ──▶  HF Space (FastAPI + retrieval)
                                                              └─▶ Claude API (answers)
                                                              └─▶ index (downloaded at boot)
```

Why this stack: the retrieval models (InLegalBERT + reranker) are CPU-friendly but
need ~2–3 GB RAM, which HF Spaces' free tier covers; Vercel is native for Next.js;
Claude removes the need for a GPU and fixes the local model's wrong-section errors.

---

## Prerequisites (your accounts)
- **GitHub** (to host the repo + the index artifact as a Release)
- **Hugging Face** account (backend host)
- **Vercel** account (frontend host)
- **Anthropic** API key with billing enabled (the answering engine)

> Heads-up: this is **outward-facing** and **costs money per question** (Claude).
> Keep `RATE_LIMIT_PER_MIN` on and set an Anthropic spend limit (see Step 5).

---

## Step 0 — Publish the prebuilt index (one-time, ~105 MB)
The index is too big for git and too slow to rebuild on the server, so the backend
downloads it at boot from `INDEX_URL`. Build it (if not already), zip it, and attach
it to a GitHub Release.

```powershell
# from nyaysetu-backend, with the venv active — only if models/index doesn't exist yet:
.\.venv\Scripts\python.exe scripts\build_index.py     # ~75 min (one-time)

# zip the CONTENTS of models/index (qdrant/ and bm25.pkl must be at the zip root):
Compress-Archive -Path models\index\* -DestinationPath nyaysetu-index.zip -Force
```

Then on GitHub: **Releases → Draft a new release → tag `index-v1` → attach
`nyaysetu-index.zip` → Publish.** Copy the asset URL — it looks like:
`https://github.com/<you>/<repo>/releases/download/index-v1/nyaysetu-index.zip`
That URL is your `INDEX_URL`.

---

## Step 1 — Backend on Hugging Face Spaces
1. **Create a Space** → SDK: **Docker** (blank), name e.g. `nyaysetu-api`, visibility Public.
2. **The Space holds only a `Dockerfile` + `README.md`** — NOT the backend code. The
   Dockerfile `git clone`s the backend from GitHub (`nyaysetu-backend/`) at build time,
   so you never copy app code into the Space. The canonical copy of those two files lives
   in **`hf-space/`** in this repo — keep them in sync. To create/update the Space, paste
   `hf-space/Dockerfile` and `hf-space/README.md` (add the Space front-matter shown in
   `hf-space/README.md`) into the Space via its **Files** tab or git push. The index is
   NOT in the image — it's fetched at boot from `INDEX_URL`.

   > **Deploy gotcha (important):** Docker caches the `RUN git clone` layer by its command
   > text, which never changes — so a plain rebuild silently re-ships the FIRST build's
   > code (symptom: `/feedback` 404 after deploying new routes). `hf-space/Dockerfile`
   > fixes this with an `ADD https://api.github.com/repos/.../commits/main …` line before
   > the clone: its content changes when `main` moves, busting the cache so every push
   > gets a fresh clone. **To ship a backend update: push to GitHub `main`, then edit the
   > Space `Dockerfile` (or Factory rebuild) to trigger a build that re-clones.**
3. **Set Space variables & secrets** (Settings → *Variables and secrets*):
   | Name | Kind | Value |
   |---|---|---|
   | `ANTHROPIC_API_KEY` | secret | `sk-ant-...` |
   | `LLM_PROVIDER` | variable | `claude` |
   | `HIGH_POWER_MODEL` | variable | `claude-haiku-4-5` |
   | `INDEX_URL` | variable | the Release URL from Step 0 |
   | `RATE_LIMIT_PER_MIN` | variable | `20` |
   | `APP_ENV` | variable | `production` |
   | `CORS_ORIGINS` | variable | *(fill in after Step 2 with the Vercel URL)* |

   **Engine choice:** `claude-haiku-4-5` is the recommended default — it hit the same
   13/13 trust-clean bar as Opus in testing at ~5–10× lower cost, the right trade-off
   for a free, at-scale tool. For maximum citation polish on hard queries, set
   `HIGH_POWER_MODEL=claude-opus-4-8` instead (same code; no other change).
4. The Space builds (first build pulls torch + caches the models; a few minutes), then
   boots and downloads the index. Verify:
   `https://<you>-nyaysetu-api.hf.space/health` → `{"status":"ok"}`, then try `/ask`.

> Free Spaces sleep after inactivity (cold start re-loads models, ~1–2 min). For
> always-on, upgrade the Space hardware (~$9/mo) or move to a small VPS (below).

---

## Step 2 — Frontend on Vercel
1. **Import the GitHub repo** → set **Root Directory = `nyaysetu-frontend`** (Vercel
   auto-detects Next.js).
2. **Environment Variables:**
   | Name | Value |
   |---|---|
   | `NEXT_PUBLIC_API_URL` | your Space URL, e.g. `https://<you>-nyaysetu-api.hf.space` |
   | `NEXT_PUBLIC_APP_NAME` | `NyaySetu` |
   | `NEXT_PUBLIC_APP_TAGLINE` | `Aapka Kanoon, Aapke Haath` |
3. **Deploy.** You get `https://<project>.vercel.app`.

---

## Step 3 — Connect them (CORS)
Set the Space's `CORS_ORIGINS` variable to your Vercel URL (comma-separated for more),
then restart the Space:
```
CORS_ORIGINS=https://<project>.vercel.app
```
Open the Vercel URL and ask a question end-to-end. Done — you're live.

---

## Step 4 — Custom domain (optional)
Vercel → Project → Domains → add your domain (free TLS). Add that domain to the
Space's `CORS_ORIGINS` too.

---

## Step 5 — Production checklist (do not skip)
- **Cap Claude spend:** set a monthly limit in the Anthropic console. Keep
  `RATE_LIMIT_PER_MIN` on (default 20/IP/min) — a public endpoint on a paid API is a
  cost/abuse target.
- **Disclaimer stays visible:** the answer disclaimer + legal-aid escalation are part
  of the trust contract; don't hide them. This is an information tool, not legal advice.
- **Watch the logs** for the first day (HF Space logs tab) — the hallucination gate
  downgrades + escalates on unverifiable citations; that's expected, not an error.
- **Cold starts:** if the sleep delay matters, upgrade to a persistent Space or VPS.

---

## Alternative backend hosts (same Dockerfile)
- **Render** — Web Service from the repo (root `nyaysetu-backend`), Docker; needs the
  2 GB plan (~$25/mo) for the models. Render injects `PORT` (already honoured).
- **Railway** — deploy the `nyaysetu-backend` Dockerfile; usage-based.
- **Fly.io** — `fly launch` in `nyaysetu-backend`; a 2 GB machine (~$10/mo), can
  scale-to-zero.
- **VPS (cheapest always-on, e.g. Hetzner CX22 ~€4.5/mo)** — install Docker, then:
  `docker build -t nyaysetu-api . && docker run -d -p 80:7860 --env-file .env.prod nyaysetu-api`
  (put `LLM_PROVIDER`, `ANTHROPIC_API_KEY`, `INDEX_URL`, `CORS_ORIGINS`,
  `RATE_LIMIT_PER_MIN` in `.env.prod`). Front it with Caddy/Nginx for TLS.

All of these read the same env vars and fetch the index via `INDEX_URL`.
