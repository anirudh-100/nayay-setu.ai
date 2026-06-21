# HF Space: `annieyyyy/nyaysetu-api` (canonical copy)

This folder mirrors **exactly** what lives in the Hugging Face Space repo
`annieyyyy/nyaysetu-api`. The Space repo holds **only** these two files — it does
**not** hold the backend code. The `Dockerfile` git-clones the backend from the
public GitHub repo (`anirudh-100/nayay-setu.ai` → `nyaysetu-backend/`) at build time.

Keep this folder in sync with the Space so the deployment is reproducible from source.

## The deploy model (and the gotcha that bit us)

The Space Dockerfile does `RUN git clone … /tmp/repo && cp -r .../nyaysetu-backend/. /app/`.
Docker caches build layers by command text, and that line never changes — so a plain
rebuild re-uses the **first build's** cloned code and silently ships stale code
(symptom: `/feedback` 404 after a deploy, follow-ups not resolving).

**Fix (already in the Dockerfile here):** an `ADD https://api.github.com/repos/anirudh-100/nayay-setu.ai/commits/main …`
line *before* the clone. Its fetched content changes whenever `main` moves, which
invalidates that layer and everything after it → a fresh clone on every new commit.

## To update the live Space
Two options, both trigger a rebuild that pulls current `main`:

1. **Web editor (simplest):** Space → Files → `Dockerfile` → edit → Commit. Any edit
   busts the cache for that build; the `ADD` line keeps future builds fresh too.
2. **Git:** clone the Space repo, copy these two files in, commit, push:
   ```bash
   git clone https://huggingface.co/spaces/annieyyyy/nyaysetu-api hf-space-live
   cp hf-space/Dockerfile hf-space/README.md hf-space-live/   # README needs the Space front-matter below
   cd hf-space-live && git add -A && git commit -m "sync Dockerfile" && git push
   ```

The Space `README.md` needs this front-matter at the top (the GitHub-facing copy here
omits it to avoid confusing repo tooling):

```
---
title: Nyaysetu Api
emoji: 🏢
colorFrom: gray
colorTo: indigo
sdk: docker
pinned: false
---
```

Then set Space **Variables & secrets** per `../DEPLOY.md` Step 1.3
(`ANTHROPIC_API_KEY`, `LLM_PROVIDER=claude`, `HIGH_POWER_MODEL=claude-haiku-4-5`,
`INDEX_URL`, `RATE_LIMIT_PER_MIN=20`, `APP_ENV=production`, `CORS_ORIGINS=<vercel url>`).
