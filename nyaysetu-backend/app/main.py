"""FastAPI entrypoint.

Startup now *loads* the persisted indexes (Qdrant + BM25) and warms the models,
instead of re-embedding the entire corpus into RAM on every boot like the old
engine did. Build the index once with ``python scripts/build_index.py``; startup is
then fast and the same index serves every request.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.routes.analyze import router as analyze_router
from app.routes.ask import router as ask_router
from app.routes.draft import router as draft_router
from app.routes.whatsapp import router as whatsapp_router
from app.services.llm_service import get_llm
from app.utils.logger import get_logger
from app.utils.rate_limit import RateLimiter

logger = get_logger(__name__)

# Per-IP cost/abuse guard on the engine endpoints (no-op when rate_limit_per_min<=0).
_rate_limiter = RateLimiter(settings.rate_limit_per_min)
# Paths that cost money (LLM) or compute — worth throttling. The webhook is excluded:
# it's signature-gated and must always 200 fast so the provider doesn't retry.
_RATE_LIMITED_PREFIXES = ("/ask", "/analyze", "/draft")


def _client_ip(request: Request) -> str:
    """Best-effort client IP, honouring the proxy header set by Vercel/HF/most PaaS."""
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _warm_engine() -> None:
    """Load models + persisted indexes into memory. Logs a clear hint if the index
    hasn't been built yet rather than crashing the whole app."""
    from app.rag.embedder import Embedder
    from app.rag.lexical_store import LexicalStore
    from app.rag.vector_store import VectorStore
    from app.utils.index_bootstrap import ensure_index

    # Production: pull the prebuilt index if it isn't on the box yet (no-op in dev).
    ensure_index()

    Embedder.instance()  # load embedding model

    try:
        store = VectorStore.instance()
        count = store.count()
        LexicalStore.instance()
        logger.info("Indexes loaded: %d vectors in Qdrant + BM25 corpus ready.", count)
    except FileNotFoundError as e:
        logger.warning("Index not found (%s). Run: python scripts/build_index.py", e)
    except Exception:
        logger.exception("Failed to load indexes at startup")

    if settings.use_reranker:
        from app.rag.reranker import Reranker

        Reranker.instance()  # load cross-encoder


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s (env=%s)", settings.app_name, settings.app_env)
    logger.info("Warming RAG engine (embedder, indexes, reranker)...")
    _warm_engine()

    logger.info("Warming up Ollama model...")
    get_llm().warmup()

    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    description="AI-powered legal assistant for India.",
    version="0.2.0",
    lifespan=lifespan,
)

# Allow the frontend (and any configured origins) to call the API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if _rate_limiter.per_minute > 0 and any(
        request.url.path.startswith(p) for p in _RATE_LIMITED_PREFIXES
    ):
        if not _rate_limiter.allow(_client_ip(request)):
            logger.warning("Rate limited %s on %s", _client_ip(request), request.url.path)
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please wait a minute and try again."},
            )
    return await call_next(request)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info("%s %s -> %d in %dms", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


@app.get("/", tags=["meta"])
def home() -> dict:
    return {"name": settings.app_name, "tagline": settings.app_tagline, "status": "ok"}


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}


app.include_router(ask_router)
app.include_router(analyze_router)
app.include_router(draft_router)
app.include_router(whatsapp_router)
