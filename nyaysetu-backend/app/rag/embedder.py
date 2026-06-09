"""Local sentence-embedding model — the dense half of hybrid retrieval.

Default model is **InLegalBERT** (``law-ai/InLegalBERT``), a BERT pre-trained on
~5.4M Indian Supreme Court & High Court documents. It understands Indian legal
vocabulary (sections, offences, procedural terms) far better than a generic
encoder like ``all-MiniLM-L6-v2``, which is the single biggest cheap win for
retrieval quality on Indian law.

Design notes:
  - InLegalBERT ships as a plain (non-sentence-transformers) BERT, so
    SentenceTransformer wraps it with **mean pooling** automatically. That is the
    correct, well-understood way to get sentence vectors out of it.
  - We **L2-normalize** every vector, so cosine similarity == dot product. The
    vector store can then use inner-product/cosine interchangeably and scores are
    bounded in [-1, 1], which makes thresholds and fusion stable.
  - Some encoders (e.g. the multilingual ``e5`` family we may switch to for Indic
    languages) require asymmetric ``query:`` / ``passage:`` prefixes. We support
    that via config so swapping models is a one-line change, not a code change.
  - Loaded **once** as a process-wide singleton — the model is the heaviest object
    in the app and is fully thread-safe for inference.

Fully local: the model downloads from Hugging Face on first run, then is cached
under ``~/.cache/huggingface`` and used offline thereafter. No API, no per-call cost.
"""
from __future__ import annotations

from threading import Lock
from typing import Sequence

import numpy as np

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Sensible default if settings hasn't been extended yet. InLegalBERT is the
# recommended default for Indian legal retrieval.
_DEFAULT_MODEL = "law-ai/InLegalBERT"


class Embedder:
    """Thread-safe singleton wrapper around a SentenceTransformer encoder."""

    _instance: "Embedder | None" = None
    _lock = Lock()

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str | None = None,
        normalize: bool = True,
        query_prefix: str = "",
        passage_prefix: str = "",
        max_seq_length: int | None = None,
    ) -> None:
        # Import lazily so merely importing this module (e.g. in tests or tooling)
        # doesn't force a multi-hundred-MB torch + model load.
        from sentence_transformers import SentenceTransformer

        self._model_name = (
            model_name
            or getattr(settings, "embedding_model", None)
            or _DEFAULT_MODEL
        )
        self._normalize = normalize
        # Asymmetric prefixes are model-specific; default to none (InLegalBERT).
        self._query_prefix = query_prefix or getattr(settings, "embedding_query_prefix", "")
        self._passage_prefix = passage_prefix or getattr(settings, "embedding_passage_prefix", "")

        logger.info("Loading embedding model %r (device=%s)", self._model_name, device or "auto")
        self._model = SentenceTransformer(self._model_name, device=device)
        if max_seq_length:
            self._model.max_seq_length = max_seq_length
        self._dim = int(self._model.get_sentence_embedding_dimension())
        logger.info("Embedding model ready: dim=%d, max_seq_len=%s", self._dim, self._model.max_seq_length)

    # ------------------------------------------------------------------ #
    # Singleton access
    # ------------------------------------------------------------------ #
    @classmethod
    def instance(cls) -> "Embedder":
        # Double-checked locking: cheap read on the hot path, lock only on first init.
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Drop the cached instance — used by tests and index rebuilds."""
        with cls._lock:
            cls._instance = None

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    # ------------------------------------------------------------------ #
    # Encoding
    # ------------------------------------------------------------------ #
    def _encode(self, texts: Sequence[str], prefix: str, batch_size: int) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        prepared = [f"{prefix}{t}" for t in texts] if prefix else list(texts)
        vectors = self._model.encode(
            prepared,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
            show_progress_bar=len(prepared) > 512,
        )
        return vectors.astype(np.float32)

    def encode_passages(self, texts: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
        """Embed documents/chunks for indexing. Shape: (n, dim)."""
        return self._encode(texts, self._passage_prefix, batch_size)

    def encode_queries(self, texts: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        """Embed user queries for search. Shape: (n, dim)."""
        return self._encode(texts, self._query_prefix, batch_size)

    def encode_query(self, text: str) -> np.ndarray:
        """Embed a single query. Shape: (dim,)."""
        return self.encode_queries([text])[0]
