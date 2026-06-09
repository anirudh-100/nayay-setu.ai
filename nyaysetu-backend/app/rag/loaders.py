"""Loaders: raw legal files → tagged :class:`Chunk` objects.

Each loader's job is not just to extract text but to attach the *metadata* that
makes a chunk citable and routable: which act, which section, whether the law is
current or repealed, and a source link. Garbage-in here caps the whole engine, so
loaders are deliberately careful (column auto-detection, cleaning, skip-and-log).

Supported sources today:
  - **IPC sections** (CSV)            → statute chunks, marked ``code_status="repealed"``
                                        (IPC was replaced by the BNS on 1 Jul 2024).
  - **IndicLegalQA** (JSON/CSV)       → qa chunks, carrying case metadata when present.
  - **Corpus guides** (.md/.txt)      → guide chunks, hierarchically chunked.

Phase 2 adds BNS/BNSS/BSA statute loaders + the IPC↔BNS mapping; the ``maps_to`` /
``code_status`` fields on Chunk already exist for exactly that.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus

import pandas as pd

from app.config import settings
from app.rag.models import Chunk
from app.utils.logger import get_logger

logger = get_logger(__name__)

_SECTION_PREFIX = re.compile(r"^(?:ipc[\s_\-]*)", re.IGNORECASE)
_BASE_SECTION = re.compile(r"^\s*(\d+[A-Za-z]?)")


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8")
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", encoding="utf-8")
    if suffix == ".json":
        return pd.read_json(path, encoding="utf-8")
    raise ValueError(f"Unsupported extension: {suffix}")


def _find_dataset(root: Path) -> Optional[Path]:
    if root.is_file():
        return root
    for ext in (".json", ".csv", ".tsv"):
        matches = sorted(root.rglob(f"*{ext}"))
        if matches:
            return matches[0]
    return None


def _clean(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def _resolve_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lower_to_actual = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_actual:
            return lower_to_actual[candidate.lower()]
    return None


def _strip_section_prefix(value: str) -> str:
    return _SECTION_PREFIX.sub("", value).strip()


# --------------------------------------------------------------------------- #
# Generic bare-act loader (registry-driven)
# --------------------------------------------------------------------------- #
# Adding a new act needs NO code: drop a CSV with a section/article column plus
# any of offence/description/punishment, and register it in data/acts/registry.json.
# This one loader replaces the old per-act IPC/BNS loaders.

def _base_section(value: str) -> str:
    m = _BASE_SECTION.match(value or "")
    return m.group(1) if m else (value or "").strip()


def _ik(query: str) -> str:
    return f"https://indiankanoon.org/search/?formInput={quote_plus(query)}"


def _act_url(template: Optional[str], *, unit: str, num: str, title: str) -> Optional[str]:
    """Build a citation link from the registry's url_template.

    - ``"ik"``            -> an Indian Kanoon search (always resolves; safe default).
    - a ``{base}`` template -> a deep link (e.g. devgan.in), using the base section.
    - ``None``            -> no link.
    """
    if not template:
        return None
    if template == "ik":
        kind = "article" if unit.lower() == "article" else "section"
        return _ik(f"{kind} {num} {title}")
    return template.replace("{base}", _base_section(num)).replace("{num}", num)


def load_act(path: Path, meta: dict) -> list[Chunk]:
    """Load one bare act from a CSV/JSON, driven by its ``registry.json`` metadata.

    Column names are auto-detected, so different sources (the Kaggle IPC dump, our
    BNS starter, hand-authored act files) all load through this one path. Penal
    provisions (those with a punishment) render as "Offence. Punishment: …"; other
    provisions fold in their descriptive text.
    """
    code = meta["code"]
    title = meta.get("title", code)
    unit = meta.get("unit", "Section")
    code_status = meta.get("code_status", "unknown")
    effective_date = meta.get("effective_date")
    url_template = meta.get("url_template")
    maps_to_act = meta.get("maps_to_act")
    is_article = unit.lower() == "article"

    df = _read_table(path)
    num_col = _resolve_col(df, ["section", "article", "number", "section_number", "section_no", "ipc_section"])
    if not num_col:
        raise KeyError(f"{code}: needs a section/article column. Got: {list(df.columns)}")
    heading_col = _resolve_col(df, ["offence", "offense", "title", "heading", "right", "name"])
    body_col = _resolve_col(df, ["description", "summary", "details", "text", "provision", "law", "definition"])
    punishment_col = _resolve_col(df, ["punishment", "penalty"])
    maps_col = _resolve_col(df, ["maps_from_ipc", "maps_from", "from_ipc"]) if maps_to_act else None

    chunks: list[Chunk] = []
    skipped = 0
    for _, row in df.iterrows():
        num = _clean(row[num_col])
        if not is_article:
            num = _strip_section_prefix(num)
        if not num:
            skipped += 1
            continue
        heading = _clean(row[heading_col]) if heading_col else ""
        body = _clean(row[body_col]) if body_col else ""
        punishment = _clean(row[punishment_col]) if punishment_col else ""
        maps_from = _clean(row[maps_col]) if maps_col else ""

        # Penal sections lead with the offence; non-penal provisions fold in the body.
        if heading:
            core = f"{heading}. {body}" if (body and body != heading and not punishment) else heading
        else:
            core = body
        if not core:
            skipped += 1
            continue
        text = f"{code} {unit} {num}: {core}"
        if punishment:
            text += f". Punishment: {punishment}"

        chunks.append(
            Chunk.create(
                text=text,
                source_type="statute",
                ref=f"{code}-{num}",
                title=title,
                act=code,
                section=None if is_article else num,
                article=num if is_article else None,
                code_status=code_status,
                effective_date=effective_date,
                maps_to=f"{maps_to_act} Section {maps_from}" if (maps_from and maps_to_act) else None,
                url=_act_url(url_template, unit=unit, num=num, title=title),
            )
        )
    logger.info("Loaded %d %s provisions from %s", len(chunks), code, path.name)
    if skipped > len(chunks) and len(df) > 3:
        logger.warning(
            "%s: skipped %d/%d rows — check %s is a well-formed CSV "
            "(quote any field containing a comma).",
            code, skipped, len(df), path.name,
        )
    return chunks


def _registry() -> list[dict]:
    path = Path(settings.data_dir) / "acts" / "registry.json"
    if not path.exists():
        logger.warning("Act registry not found at %s", path)
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("acts", [])


def load_acts() -> list[Chunk]:
    """Load every act listed in ``data/acts/registry.json``. Missing files are skipped
    with a warning. IPC chunks get their BNS successor back-filled via the mapping."""
    root = Path(settings.data_dir).resolve().parent
    chunks: list[Chunk] = []
    for meta in _registry():
        rel = meta.get("path")
        if not rel:
            continue
        path = Path(rel)
        if not path.is_absolute():
            path = root / rel
        if path.is_dir():
            path = _find_dataset(path)
        if not path or not path.exists():
            logger.warning("Act %s: data file not found (%s)", meta.get("code"), rel)
            continue
        try:
            chunks += load_act(path, meta)
        except Exception:
            logger.exception("Failed to load act %s from %s", meta.get("code"), path)

    _link_ipc_to_bns([c for c in chunks if c.act == "IPC"])
    return chunks


# --------------------------------------------------------------------------- #
# IndicLegalQA loader
# --------------------------------------------------------------------------- #
def load_indiclegalqa(path: Path) -> list[Chunk]:
    """IndicLegalQA → qa chunks, carrying case metadata (name/court/date) when present."""
    df = _read_table(path)
    q_col = _resolve_col(df, ["question"])
    a_col = _resolve_col(df, ["answer"])
    if not q_col or not a_col:
        raise KeyError(f"QA dataset needs question+answer columns. Got: {list(df.columns)}")
    case_col = _resolve_col(df, ["case_name", "case", "title"])
    court_col = _resolve_col(df, ["court"])
    date_col = _resolve_col(df, ["judgement_date", "judgment_date", "date"])

    chunks: list[Chunk] = []
    for i, row in df.iterrows():
        question = _clean(row[q_col])
        answer = _clean(row[a_col])
        if not question or not answer:
            continue
        case_name = _clean(row[case_col]) if case_col else ""
        court = _clean(row[court_col]) if court_col else ""
        date = _clean(row[date_col]) if date_col else ""

        text = f"Question: {question} Answer: {answer}"
        chunks.append(
            Chunk.create(
                text=text,
                source_type="qa",
                ref=f"qa-{i}-{case_name or question[:40]}",
                title=case_name or None,
                court=court or None,
                effective_date=date or None,
                code_status="unknown",
            )
        )
    logger.info("Loaded %d IndicLegalQA pairs from %s", len(chunks), path.name)
    return chunks


# --------------------------------------------------------------------------- #
# Corpus guide loader (markdown / text) with hierarchical chunking
# --------------------------------------------------------------------------- #
def _chunk_markdown(text: str, *, max_words: int, overlap: int) -> list[str]:
    """Split on Markdown headings first (keeps logical sections intact), then
    window any over-long section by words with overlap. Naive fixed-size chunking
    severs legal context; heading-aware chunking preserves it."""
    # Split into (heading-led) sections.
    sections = re.split(r"(?m)^(?=#{1,6}\s)", text)
    sections = [s.strip() for s in sections if s.strip()]
    if not sections:
        sections = [text.strip()]

    out: list[str] = []
    for section in sections:
        words = section.split()
        if len(words) <= max_words:
            out.append(section)
            continue
        step = max(1, max_words - overlap)
        for start in range(0, len(words), step):
            piece = " ".join(words[start : start + max_words])
            if piece.strip():
                out.append(piece)
            if start + max_words >= len(words):
                break
    return out


def load_corpus(corpus_dir: Path, *, max_words: int, overlap: int) -> list[Chunk]:
    """Load .md/.txt plain-language legal guides into hierarchically-chunked guide chunks."""
    chunks: list[Chunk] = []
    if not corpus_dir.exists():
        return chunks
    for path in sorted(corpus_dir.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            continue
        # Title from first H1 if present, else filename.
        h1 = re.search(r"(?m)^#\s+(.+)$", raw)
        title = h1.group(1).strip() if h1 else path.stem.replace("_", " ").title()
        for j, piece in enumerate(_chunk_markdown(raw, max_words=max_words, overlap=overlap)):
            chunks.append(
                Chunk.create(
                    text=piece,
                    source_type="guide",
                    ref=f"{path.stem}-{j}",
                    title=title,
                    code_status="unknown",
                )
            )
    logger.info("Loaded %d guide chunks from %s", len(chunks), corpus_dir)
    return chunks


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
def load_all() -> list[Chunk]:
    """Load every configured source into one chunk list. Missing sources are skipped
    with a warning rather than failing the whole build."""
    chunks: list[Chunk] = []

    data_dir = Path(settings.data_dir)
    qa_dir = Path(getattr(settings, "qa_dir", data_dir / "indiclegalqa"))
    corpus_dir = Path(settings.corpus_dir)
    max_words = int(getattr(settings, "chunk_size", 700))
    overlap = int(getattr(settings, "chunk_overlap", 100))

    # All bare acts come from the registry (IPC, BNS, and any added acts).
    chunks += load_acts()

    qa_path = _find_dataset(qa_dir) if qa_dir.exists() else None
    if qa_path:
        chunks += load_indiclegalqa(qa_path)
    else:
        logger.warning("IndicLegalQA dataset not found under %s", qa_dir)

    chunks += load_corpus(corpus_dir, max_words=max_words, overlap=overlap)

    logger.info("Loaded %d total chunks across all sources", len(chunks))
    return chunks


def _link_ipc_to_bns(ipc_chunks: list[Chunk]) -> None:
    """Set ``maps_to`` on IPC chunks using the curated IPC<->BNS mapping."""
    from app.rag.law_map import LawMap

    law_map = LawMap.instance()
    linked = 0
    for c in ipc_chunks:
        if not c.section:
            continue
        entry = law_map.bns_for_ipc(c.section)
        if entry and entry.get("bns"):
            c.maps_to = f"BNS Section {entry['bns']}"
            linked += 1
    logger.info("Linked %d/%d IPC chunks to their BNS successor", linked, len(ipc_chunks))
