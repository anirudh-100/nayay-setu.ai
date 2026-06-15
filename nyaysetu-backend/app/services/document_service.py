"""Document understanding: explain a legal document a citizen received, grounded in law.

This is the "Understand my document" capability — the citizen-facing counterpart to a
lawyer reading your notice and telling you what it means. The flow mirrors the trust
contract of the Q&A engine so it inherits the same guarantees:

    document text
      -> retrieve the most relevant law from the corpus (hybrid retriever)
      -> ask the LLM to explain the document USING ONLY that retrieved law
      -> verify any cited section was actually retrieved (hallucination gate)
      -> attach citations + current-law note + disclaimer; escalate if unsure

The model is told to explain, not to render a definitive legal opinion, and every legal
reference is grounded in a retrieved source — so the same "cite, don't hallucinate"
discipline that protects the /ask path protects document analysis too.
"""
from __future__ import annotations

import time
from typing import Optional

from app.rag.models import Citation, RetrievedChunk
from app.rag.retriever import HybridRetriever
from app.schemas.analyze import AnalyzeResponse
from app.schemas.ask import DISCLAIMER, LEGAL_AID_ESCALATION, Confidence
from app.services.llm_service import OllamaClient, get_llm
from app.services.rag_service import (
    _format_context,
    _order_for_context,
    _stringify,
    _verify_citation,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

_CONTEXT_CHUNKS = 6
_MAX_CITATIONS = 5
# How much of the document to put in the prompt / use as the retrieval query. The
# embedder truncates internally, but we cap the prompt to keep latency/cost bounded.
_PROMPT_DOC_CHARS = 6000
_QUERY_DOC_CHARS = 2000


PROMPT_TEMPLATE = """You are an AI legal assistant for India. A person has shared a legal document they received and wants to understand it in plain language.

---
DOCUMENT (verbatim; may be truncated):
{document}
{question_block}
---
RELEVANT INDIAN LAW (retrieved sources; you may cite ONLY these, by their section/article):
{context}

---
TASK: Explain this document to an ordinary person in simple, calm language.
RULES:
- "document_type": what this document is, in plain words (e.g. "Police summons", "Legal notice for cheque bounce", "Rent agreement").
- "summary": 2-4 sentences on what it says and what it means for the person.
- "key_points": array of the most important things to understand (short strings).
- "deadlines": array of any dates or time-limits the person must act by — quote the date/period from the document. Use [] if there are none.
- "action": the single most important next step.
- "law_references": array of section labels FROM THE RELEVANT LAW above that apply (e.g. "BNS Section 318"). Use [] if none clearly apply. Do NOT invent section numbers that are not in the RELEVANT INDIAN LAW above.
- Prefer CURRENT law (BNS / BNSS / BSA) over repealed law (IPC / CrPC / Evidence Act) when both appear.
- This is information to help them understand — do NOT give a definitive legal opinion.

Return ONLY this JSON object (no prose before or after):
{{
  "document_type": "...",
  "summary": "...",
  "key_points": ["..."],
  "deadlines": ["..."],
  "action": "...",
  "law_references": ["..."]
}}
"""


def _stringify_list(value: object) -> list[str]:
    """Coerce an LLM field into a clean list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        v = value.strip()
        return [v] if v else []
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            s = _stringify(item)
            if s:
                out.append(s)
        return out
    s = _stringify(value)
    return [s] if s else []


class DocumentService:
    def __init__(
        self,
        llm: OllamaClient | None = None,
        retriever: HybridRetriever | None = None,
    ) -> None:
        self._llm = llm or get_llm()
        self._retriever = retriever or HybridRetriever()

    def analyze(self, document_text: str, question: str | None = None, language: str = "en") -> AnalyzeResponse:
        started = time.perf_counter()
        document_text = document_text.strip()

        # 1. Retrieve the law most relevant to the document (and the user's question).
        query = f"{question or ''} {document_text[:_QUERY_DOC_CHARS]}".strip()
        results = self._retriever.retrieve(query)
        ordered = _order_for_context(results)
        context = _format_context(ordered[:_CONTEXT_CHUNKS]) if ordered else "(no closely matching law found)"

        # 2. Ask the LLM to explain the document using only that law.
        question_block = f"\nTHE PERSON ALSO ASKS: {question}\n" if question else "\n"
        prompt = PROMPT_TEMPLATE.format(
            document=document_text[:_PROMPT_DOC_CHARS],
            question_block=question_block,
            context=context,
        )
        raw = self._llm.generate_json(prompt)

        document_type = _stringify(raw.get("document_type")) or "Legal document"
        summary = _stringify(raw.get("summary")) or (
            "This appears to be a legal document. I couldn't fully classify it from the text provided."
        )
        key_points = _stringify_list(raw.get("key_points"))
        deadlines = _stringify_list(raw.get("deadlines"))
        action = _stringify(raw.get("action")) or (
            "Consider consulting a lawyer or free legal aid to understand your options and any time limits."
        )
        law_references = _stringify_list(raw.get("law_references"))

        # 3. Hallucination gate: every cited section must appear in the retrieved law.
        citation_verified = all(_verify_citation(ref, ordered) for ref in law_references) if law_references else True
        if not citation_verified:
            logger.warning("Document analysis cited unretrieved law %r — downgrading.", law_references)

        # 4. Deterministic confidence from how well it was grounded.
        if not ordered:
            confidence: Confidence = "medium"  # can explain the document, but no specific law matched
        elif not citation_verified:
            confidence = "low"
        elif law_references:
            confidence = "high"
        else:
            confidence = "medium"

        current_law_note = self._current_law_note(ordered)
        citations = self._dedupe_citations(ordered)
        escalation = LEGAL_AID_ESCALATION if (confidence == "low" or not citation_verified) else None

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "Analyzed document (%d chars) in %dms | type=%r | confidence=%s | verified=%s",
            len(document_text),
            elapsed_ms,
            document_type,
            confidence,
            citation_verified,
        )

        return AnalyzeResponse(
            document_type=document_type,
            summary=summary,
            key_points=key_points,
            deadlines=deadlines,
            action=action,
            confidence=confidence,
            citations=citations,
            current_law_note=current_law_note,
            citation_verified=citation_verified,
            abstained=False,
            escalation=escalation,
            disclaimer=DISCLAIMER,
            response_time_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------ #
    def _current_law_note(self, results: list[RetrievedChunk]) -> Optional[str]:
        """Bridge the top repealed section (IPC/CrPC/IEA) to its current successor."""
        from app.rag.law_map import LawMap

        law_map = LawMap.instance()
        from_codes = set(law_map.from_codes())
        for rc in results:
            c = rc.chunk
            if c.act in from_codes and c.section:
                note = law_map.current_reference_note(c.act, c.section)
                if note:
                    if not law_map.verified_for(c.act):
                        note += " (Mapping is indicative — confirm against the official bare act.)"
                    return note
        return None

    @staticmethod
    def _dedupe_citations(results: list[RetrievedChunk]) -> list[Citation]:
        seen: set[str] = set()
        out: list[Citation] = []
        for rc in results:
            cit = rc.to_citation()
            if cit.label in seen:
                continue
            seen.add(cit.label)
            out.append(cit)
            if len(out) >= _MAX_CITATIONS:
                break
        return out
