"""Application configuration (env-driven via pydantic-settings).

Every tunable for the RAG engine lives here so the pipeline can be tuned without
code edits — set any field in ``.env`` (UPPER_SNAKE_CASE). Branding is config-driven
on purpose (``app_name``) so the product can be renamed without touching code.
"""
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App / branding (neutral, swappable) ---
    app_name: str = "NyaySetu"
    app_tagline: str = "Aapka Kanoon, Aapke Haath"
    app_env: str = "development"
    log_level: str = "INFO"
    # Comma-separated origins allowed to call the API (the frontend dev server, etc.).
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # --- LLM (Ollama, local) ---
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "mistral"
    ollama_timeout_s: int = 300

    # --- Embeddings (dense retrieval) ---
    # InLegalBERT understands Indian legal text far better than a generic encoder.
    embedding_model: str = "law-ai/InLegalBERT"
    # Asymmetric prefixes for e5-style models; empty for InLegalBERT.
    embedding_query_prefix: str = ""
    embedding_passage_prefix: str = ""

    # --- Reranker (precision pass) ---
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_reranker: bool = True

    # --- Vector store (Qdrant) ---
    # Leave qdrant_url empty to run embedded (local file). Set it to a server URL in prod.
    qdrant_url: str = ""
    qdrant_collection: str = "nyaysetu_chunks"

    # --- Retrieval tuning ---
    fetch_k: int = 30          # candidates pulled from EACH retriever before fusion
    rerank_candidates: int = 50  # fused shortlist size handed to the reranker
    top_k: int = 6             # chunks finally shown to the LLM
    rrf_k: int = 60            # RRF damping constant
    # Below this top rerank score, the engine abstains instead of risking a
    # hallucinated section. Permissive default; raise it for stricter abstention.
    min_rerank_score: float = -10.0

    # --- Data + index locations ---
    data_dir: Path = Field(default=DATA_DIR)
    corpus_dir: Path = Field(default=DATA_DIR / "corpus")
    ipc_dir: Path = Field(default=DATA_DIR / "ipc")
    bns_dir: Path = Field(default=DATA_DIR / "bns")
    qa_dir: Path = Field(default=DATA_DIR / "indiclegalqa")
    index_dir: Path = Field(default=ROOT_DIR / "models" / "index")

    # --- Chunking ---
    chunk_size: int = 700      # max words per guide chunk
    chunk_overlap: int = 100   # word overlap between consecutive chunks

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def qdrant_path(self) -> Path:
        """Filesystem path for embedded Qdrant (used when qdrant_url is empty)."""
        return self.index_dir / "qdrant"

    @property
    def bm25_file(self) -> Path:
        return self.index_dir / "bm25.pkl"


settings = Settings()
