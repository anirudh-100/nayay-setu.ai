# NyaySetu backend — FastAPI + InLegalBERT retrieval + Claude answering (CPU only).
# Build context = this directory (nyaysetu-backend/). Works on Hugging Face Spaces
# (Docker SDK), Render, Railway, Fly.io, or any Docker host.
#
# The 105 MB index is NOT baked in: it's fetched at boot from INDEX_URL (see
# app/utils/index_bootstrap.py), keeping the image small and the repo light.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    PORT=7860

WORKDIR /app

# Install CPU-only torch first so sentence-transformers doesn't pull the (huge) CUDA build.
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt

# Pre-cache the embedding + reranker models so first boot is fast and works offline.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('law-ai/InLegalBERT'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); print('models cached')"

# App code + data (the small statute/guide CSVs). The index arrives at boot via INDEX_URL.
COPY . .

EXPOSE 7860
# Honour $PORT (Render/Railway inject it); default 7860 for Hugging Face Spaces.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
