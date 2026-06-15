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

    # --- LLM (Ollama, local) — the free default engine ---
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "mistral"
    ollama_timeout_s: int = 300

    # --- High-power mode (optional, OFF by default) ---
    # Which engine answers: "ollama" (free, local) or "claude" (cloud, higher quality).
    # Stays "ollama" unless explicitly switched, so the free/local default is untouched.
    llm_provider: str = "ollama"
    # Required only when llm_provider="claude". Empty => high-power mode unavailable.
    anthropic_api_key: str = ""
    # Opus 4.8 is the most capable tier — best for nuanced, current Indian law.
    high_power_model: str = "claude-opus-4-8"
    anthropic_timeout_s: int = 120

    # --- Messaging channels (Pillar 4, OFF by default) ---
    # How replies are delivered: "console" (local, logs only, nothing sent) or
    # "whatsapp" (Meta Cloud API). Stays "console" so nothing leaves the machine.
    messaging_provider: str = "console"
    # WhatsApp Cloud API credentials — required only when messaging_provider="whatsapp".
    whatsapp_verify_token: str = ""     # your chosen token for the webhook handshake
    whatsapp_app_secret: str = ""       # Meta app secret; verifies inbound signatures
    whatsapp_phone_number_id: str = ""  # the Cloud API phone number id (sender)
    whatsapp_access_token: str = ""      # Graph API access token
    whatsapp_graph_url: str = "https://graph.facebook.com/v21.0"
    whatsapp_timeout_s: int = 30

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
    # hallucinated section. 0.0 = abstain when even the best-matched law scores as
    # net-irrelevant (a negative cross-encoder logit). Calibrated on real queries
    # (scripts/probe_abstain_scores.py): genuine legal questions scored +2.7..+9.4,
    # off-topic ones -11.1..-1.6 — so 0.0 sits in the clean gap with margin both ways.
    min_rerank_score: float = 0.0

    # --- Deployment (production) ---
    # When the index is missing at startup and this is set, the backend downloads a
    # prebuilt index zip from here (e.g. a GitHub Release) instead of rebuilding it on
    # the box (~75 min). Empty in dev, where the locally built index is used.
    index_url: str = ""
    # Per-IP requests/minute cap on the answering endpoints (protects the paid Claude
    # API from abuse / runaway cost). 0 disables it (the dev default).
    rate_limit_per_min: int = 0

    # --- Data + index locations ---
    data_dir: Path = Field(default=DATA_DIR)
    corpus_dir: Path = Field(default=DATA_DIR / "corpus")
    ipc_dir: Path = Field(default=DATA_DIR / "ipc")
    bns_dir: Path = Field(default=DATA_DIR / "bns")
    qa_dir: Path = Field(default=DATA_DIR / "indiclegalqa")
    judgments_dir: Path = Field(default=DATA_DIR / "judgments")
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
