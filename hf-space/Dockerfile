FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf-cache \
    PORT=7860

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# Cache-bust: this fetch's content changes whenever GitHub main moves, so the
# git clone below is NEVER served from a stale Docker layer. Without it, Docker
# caches the `RUN git clone` layer (its command text never changes) and every
# rebuild silently re-ships the FIRST build's code — the bug that kept /feedback
# returning 404 after deploys.
ADD https://api.github.com/repos/anirudh-100/nayay-setu.ai/commits/main /tmp/main-commit.json

RUN git clone --depth 1 https://github.com/anirudh-100/nayay-setu.ai /tmp/repo \
 && cp -r /tmp/repo/nyaysetu-backend/. /app/ \
 && rm -rf /tmp/repo

RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('law-ai/InLegalBERT'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

RUN mkdir -p /app/models/index /app/hf-cache && chmod -R 777 /app/models /app/hf-cache

EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
