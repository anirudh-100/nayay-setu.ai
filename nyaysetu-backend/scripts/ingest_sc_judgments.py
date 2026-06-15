"""Ingest a curated set of recent Supreme Court judgments into data/judgments/.

Source: the official **Indian Supreme Court Judgments** open dataset on AWS
(s3://indian-supreme-court-judgments, CC-BY-4.0, sourced from eCourts; managed by Dattam
Labs). Everything is read over plain HTTPS — no AWS credentials/CLI needed.

Pipeline (deterministic — no LLM ever touches the judgment text):
  1. download the per-year metadata parquet (title, citation, decision_date, court, path)
  2. for each candidate, fetch the English PDF (path -> ``<path>_EN.pdf``)
  3. extract the text layer with pypdf; QUALITY-GATE it (length + alphabetic ratio +
     legal-vocabulary check) so scanned/garbled PDFs are skipped (we have no OCR)
  4. clean + cap each judgment to keep the index bounded, and write a CSV that
     ``app.rag.loaders.load_judgments`` ingests as ``judgment`` chunks.

The text is verbatim from the official PDF (not paraphrased), but it's bulk auto-extracted
and not individually proofread, so it's honestly marked ``verification="unverified"`` (per
the strict official/curated/unverified contract) with the official source in source_authority.

Usage:
    python scripts/ingest_sc_judgments.py            # default: ~40 from 2024
    python scripts/ingest_sc_judgments.py 2024 2023 60
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BUCKET = "https://indian-supreme-court-judgments.s3.amazonaws.com"
RAW = Path("data/_raw_sources")
PDF_CACHE = RAW / "sc_pdfs"
OUT = Path("data/judgments/sc_judgments.csv")

DEFAULT_YEARS = [2024]
DEFAULT_TARGET = 40
MAX_WORDS = 8000          # cap per judgment so one opinion can't dominate the index
MIN_CHARS = 1800          # below this, the PDF is almost certainly scanned/empty
MIN_ALPHA_RATIO = 0.6     # garbled extractions have lots of non-letters
_LEGAL_WORDS = ("court", "appeal", "judgment", "order", "petition", "section", "honourable", "learned")

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NyaySetu-research"


def _client():
    import httpx

    return httpx.Client(timeout=60, headers={"User-Agent": UA}, follow_redirects=True)


def _download(client, url: str, dst: Path) -> bool:
    if dst.exists() and dst.stat().st_size > 0:
        return True
    try:
        r = client.get(url)
        if r.status_code != 200 or not r.content:
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(r.content)
        return True
    except Exception:
        return False


def _clean(text: str) -> str:
    text = text.replace("�", "")            # drop replacement chars (mojibake)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    # Drop lines that are just page numbers / digital-signature boilerplate.
    keep = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if re.fullmatch(r"\d{1,4}", s):
            continue
        if re.search(r"(?i)digitally signed|signature not verified|date:\s*\d{4}\.\d", s):
            continue
        keep.append(s)
    return re.sub(r"\s+", " ", " ".join(keep)).strip()


def _looks_like_text(text: str) -> bool:
    if len(text) < MIN_CHARS:
        return False
    letters = sum(ch.isalpha() for ch in text)
    nonspace = sum(not ch.isspace() for ch in text) or 1
    if letters / nonspace < MIN_ALPHA_RATIO:
        return False
    low = text.lower()
    return sum(w in low for w in _LEGAL_WORDS) >= 2


def _cap_words(text: str, n: int) -> str:
    """Cap to ~n words, but trim back to the last sentence end so a capped judgment
    doesn't end mid-sentence (which would read as a misleading partial holding)."""
    words = text.split()
    if len(words) <= n:
        return text
    capped = " ".join(words[:n])
    cut = max(capped.rfind(". "), capped.rfind("? "), capped.rfind("! "))
    if cut > len(capped) * 0.7:  # only trim if a sentence end is reasonably near the cap
        capped = capped[: cut + 1]
    return capped.strip()


def _extract_pdf(path: Path) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception:
        return ""


def main(argv: list[str]) -> int:
    import pandas as pd

    years = [int(a) for a in argv if a.isdigit() and len(a) == 4] or DEFAULT_YEARS
    target = next((int(a) for a in argv if a.isdigit() and len(a) != 4), DEFAULT_TARGET)
    print(f"Ingesting up to {target} SC judgments from year(s) {years} (text-layer only)\n" + "=" * 72)

    rows_out: list[dict] = []
    seen_titles: set[str] = set()
    with _client() as client:
        for year in years:
            if len(rows_out) >= target:
                break
            pq = RAW / f"sc_meta_{year}.parquet"
            if not _download(client, f"{BUCKET}/metadata/parquet/year={year}/metadata.parquet", pq):
                print(f"  [skip] could not fetch metadata parquet for {year}")
                continue
            df = pd.read_parquet(pq)
            # Stable order; prefer rows that look substantive (have a citation + path).
            df = df[df["path"].astype(str).str.len() > 0].sort_values("citation").reset_index(drop=True)
            print(f"  {year}: {len(df)} judgments in metadata; scanning for clean text layers...")

            scanned = kept = skipped_scanned = 0
            for _, row in df.iterrows():
                if len(rows_out) >= target:
                    break
                path = str(row.get("path", "")).strip()
                title = re.sub(r"\s+", " ", str(row.get("title", "")).replace("�", "")).strip()
                if not path or not title or title in seen_titles:
                    continue
                langs = str(row.get("available_languages", ""))
                if "ENG" not in langs.upper():
                    continue

                scanned += 1
                pdf_url = f"{BUCKET}/data/pdf/year={year}/english/{path}_EN.pdf"
                pdf_dst = PDF_CACHE / f"{path}_EN.pdf"
                if not _download(client, pdf_url, pdf_dst):
                    continue
                text = _clean(_extract_pdf(pdf_dst))
                if not _looks_like_text(text):
                    skipped_scanned += 1
                    continue

                seen_titles.add(title)
                rows_out.append({
                    "case_name": title,
                    "court": str(row.get("court", "") or "Supreme Court of India").strip(),
                    "citation": str(row.get("citation", "")).strip(),
                    "decision_date": str(row.get("decision_date", "")).strip(),
                    "text": _cap_words(text, MAX_WORDS),
                    # Honest trust level: the SOURCE is the official SC judgment, but the
                    # text is bulk auto-extracted and not individually proofread/verified —
                    # so it's "unverified" (per the strict official/curated/unverified
                    # contract), with the official provenance carried in source_authority.
                    "verification": "unverified",
                    "source_authority": "Supreme Court of India judgment, official PDF via AWS Open Data / eCourts "
                                        "(text auto-extracted, not individually proofread)",
                    "official_url": pdf_url,
                })
                kept += 1
                if kept % 10 == 0:
                    print(f"    ...{kept} kept")
            print(f"  {year}: scanned {scanned}, kept {kept}, skipped {skipped_scanned} (scanned/garbled)")

    if not rows_out:
        print("FAIL: no judgments ingested (network or all-scanned?)")
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case_name", "court", "citation", "decision_date", "text",
              "verification", "source_authority", "official_url"]
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)

    avg_words = sum(len(r["text"].split()) for r in rows_out) // len(rows_out)
    print("=" * 72)
    print(f"[OK] wrote {len(rows_out)} judgments to {OUT}  (avg {avg_words} words each)")
    print("Samples:")
    for r in rows_out[:3]:
        print(f"  - {r['case_name'][:60]}  | {r['citation']} | {r['decision_date']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
