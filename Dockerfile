# Root Dockerfile for one-click PaaS deploys that build from the repo root (e.g. Railway),
# so no monorepo "root directory" setting is needed. It builds the backend in
# nyaysetu-backend/. (nyaysetu-backend/Dockerfile is the canonical one for hosts where you
# CAN set a service root — e.g. Hugging Face Spaces; this mirrors it from the repo root.)
#
# CPU-only. The 105 MB index is NOT baked in — it's fetched at boot from INDEX_URL
# (see nyaysetu-backend/app/utils/index_bootstrap.py).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    PORT=8080

WORKDIR /app

# Install CPU-only torch first so sentence-transformers doesn't pull the (huge) CUDA build.
COPY nyaysetu-backend/requirements.txt ./
RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt

# Pre-cache the embedding + reranker models so first boot is fast and works offline.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('law-ai/InLegalBERT'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); print('models cached')"

# Backend app code + data (the small statute/guide CSVs). Index arrives at boot via INDEX_URL.
COPY nyaysetu-backend/ ./

EXPOSE 8080
# Honour $PORT (Railway/Render inject it); default 8080.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
