"""Core data contract for the RAG engine.

Everything downstream — the vector store payloads, the lexical (BM25) index, the
reranker, and the user-facing citations — is built on these models. The guiding
idea: a chunk is never *just* text. It always carries the legal metadata needed to
(a) filter retrieval (by act, jurisdiction, date, language, code status) and
(b) produce a *verifiable* citation the user can click through to the source.

Why this matters for Indian law specifically:
  - ``act`` + ``section``/``article`` let us cite "BNS Section 318" precisely.
  - ``code_status`` + ``maps_to`` support the IPC↔BNS transition (offences before
    1 Jul 2024 use the old codes; we can surface both old and new references).
  - ``effective_date`` enables date-aware routing later (which law applied when).
  - ``language`` keeps the door open for multilingual retrieval.
"""
from __future__ import annotations

import hashlib
import re
from typing import Literal, Optional
from urllib.parse import quote_plus

from pydantic import BaseModel, Field

_BASE_SECTION_RE = re.compile(r"(\d+[A-Za-z]?)")


def _ik_search(query: str) -> str:
    """A reliable Indian Kanoon search link (always resolves, never 500s)."""
    return f"https://indiankanoon.org/search/?formInput={quote_plus(query)}"

# What kind of legal source a chunk came from. Drives prompting, citation style,
# and (later) per-type chunking strategies.
SourceType = Literal["statute", "judgment", "qa", "guide"]

# Whether the underlying law is currently in force. Lets us prefer current law
# (BNS/BNSS/BSA) while still answering about repealed law (IPC/CrPC) for older
# incidents. ``unknown`` is the safe default for sources we haven't classified.
CodeStatus = Literal["current", "repealed", "unknown"]

# How trustworthy the *text itself* is — the heart of the product's promise. This is
# distinct from ``code_status`` (which law applies) and is surfaced on every citation
# so a user can see, at a glance, whether to rely on a source or confirm it first:
#   - ``official``   — text taken from an authoritative source (e.g. India Code) and
#                      verified against it. Safe to rely on.
#   - ``curated``    — hand-compiled starter / plain-language guide. Useful, but flagged
#                      "confirm before relying" until replaced with an official source.
#   - ``unverified`` — bulk-ingested with no per-item check (e.g. a public QA dataset).
# A wrong-but-confident citation is the exact failure this engine exists to prevent, so
# we never silently present ``curated``/``unverified`` text as if it were ``official``.
Verification = Literal["official", "curated", "unverified"]


def make_chunk_id(source_type: str, ref: str, text: str) -> str:
    """Deterministic, content-addressed id.

    Stable across rebuilds (so re-ingesting doesn't duplicate vectors) yet
    sensitive to text changes (so edited content gets a fresh id). ``ref`` is a
    human-meaningful key (e.g. "BNS-318" or a case name) to keep ids debuggable.
    """
    digest = hashlib.sha1(f"{source_type}|{ref}|{text}".encode("utf-8")).hexdigest()
    return f"{source_type}-{digest[:16]}"


class Chunk(BaseModel):
    """One retrievable unit of legal text plus its provenance.

    Stored verbatim as the vector-store payload and indexed by the BM25 store, so
    keep it JSON-serializable and reasonably small.
    """

    id: str
    text: str
    source_type: SourceType

    # --- Provenance / citation fields (all optional; populated per source) ---
    title: Optional[str] = None          # e.g. "Bharatiya Nyaya Sanhita, 2023"
    act: Optional[str] = None            # short code, e.g. "BNS", "IPC", "CrPC", "BNSS"
    section: Optional[str] = None        # e.g. "318"
    article: Optional[str] = None        # Constitution articles, e.g. "21"
    court: Optional[str] = None          # judgments, e.g. "Supreme Court of India"
    case_citation: Optional[str] = None  # e.g. "(2017) 10 SCC 1"
    effective_date: Optional[str] = None  # ISO date: law-in-force or judgment date
    url: Optional[str] = None            # canonical link for click-through

    # --- Retrieval / routing metadata ---
    jurisdiction: str = "India"
    language: str = "en"
    code_status: CodeStatus = "unknown"
    maps_to: Optional[str] = None        # cross-ref, e.g. IPC 420 -> "BNS 318"

    # --- Provenance / trust (surfaced on every citation) ---
    # Default to the safe, honest value: assume nothing is verified until a loader says so.
    verification: Verification = "unverified"
    source_authority: Optional[str] = None  # e.g. "India Code (indiacode.nic.in)"
    official_url: Optional[str] = None       # authoritative source, distinct from the click-through
    retrieved_at: Optional[str] = None       # ISO date the text was pulled from its source

    def source_url(self) -> Optional[str]:
        """A clickable link that actually resolves — computed canonically by source.

        Computed (not just read from ``url``) on purpose, so it repairs links across
        the whole index without a rebuild:
          - **BNS** -> devgan.in section page (verified working).
          - **IPC** -> Indian Kanoon search (devgan's IPC pages are broken/deprecated
            since the IPC's repeal, so a stable search link beats a dead 500 page).
          - other statutes/guides -> their stored ``url`` if any.
          - **case Q&A / judgments** -> Indian Kanoon search by case name/citation.
        """
        base = ""
        if self.section:
            m = _BASE_SECTION_RE.search(self.section)
            base = m.group(1) if m else ""

        if self.act == "BNS" and base:
            return f"https://devgan.in/bns/section/{base}/"
        if self.act == "IPC" and self.section:
            return _ik_search(f"section {self.section} indian penal code")
        if self.url:
            return self.url
        if self.source_type in ("qa", "judgment"):
            key = self.case_citation or self.title
            if key:
                return _ik_search(key)
        return None

    def reference_label(self) -> str:
        """Short human label for citations, e.g. 'BNS Section 318' or a case name."""
        if self.act and self.section:
            return f"{self.act} Section {self.section}"
        if self.article:
            return f"Constitution Article {self.article}"
        if self.case_citation:
            return self.case_citation
        if self.court and self.title:
            return f"{self.title} ({self.court})"
        return self.title or self.source_type.upper()

    @classmethod
    def create(cls, *, text: str, source_type: SourceType, ref: str, **kwargs) -> "Chunk":
        """Build a chunk with a deterministic id derived from ``ref`` + ``text``."""
        return cls(id=make_chunk_id(source_type, ref, text), text=text, source_type=source_type, **kwargs)


class Citation(BaseModel):
    """A verifiable source pointer surfaced to the user alongside an answer."""

    label: str                       # "BNS Section 318"
    source_type: SourceType
    snippet: str                     # the supporting text, trimmed
    url: Optional[str] = None
    code_status: CodeStatus = "unknown"

    # --- Trust signal (drives the "Official / Curated / Unverified" badge in the UI) ---
    verification: Verification = "unverified"
    source_authority: Optional[str] = None

    @classmethod
    def from_chunk(cls, chunk: "Chunk", *, snippet_chars: int = 320) -> "Citation":
        snippet = chunk.text.strip()
        if len(snippet) > snippet_chars:
            snippet = snippet[: snippet_chars - 1].rstrip() + "…"
        return cls(
            label=chunk.reference_label(),
            source_type=chunk.source_type,
            snippet=snippet,
            url=chunk.source_url(),
            code_status=chunk.code_status,
            verification=chunk.verification,
            source_authority=chunk.source_authority,
        )


class RetrievedChunk(BaseModel):
    """A chunk returned by retrieval, with the scores that got it there.

    Keeping the component scores (not just the final one) makes the hybrid
    pipeline debuggable and lets us tune fusion/reranking with real numbers.
    """

    chunk: Chunk
    score: float = 0.0                       # final score after fusion/rerank
    dense_score: Optional[float] = None      # cosine similarity from vector search
    lexical_score: Optional[float] = None    # BM25 score
    rerank_score: Optional[float] = None     # cross-encoder relevance

    def to_citation(self) -> Citation:
        return Citation.from_chunk(self.chunk)
