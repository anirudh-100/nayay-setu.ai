"""RAG orchestrator: hybrid retrieve → current-law expansion → grounded, verified answer.

Phase 1 gave us hybrid retrieval + citations + abstention. Phase 2 makes the answers
*current and accountable* — the things that make a legal tool worth trusting:

  - **Cross-reference expansion** — when retrieval surfaces a repealed IPC section, we
    pull its current BNS successor into context too, so the answer leads with law that
    is actually in force (not colonial-era law presented as current).
  - **Current-law note** — every IPC reference is bridged to its BNS equivalent with the
    1 July 2024 transition rule, so users who only know the old number aren't misled.
  - **Citation verification** — after generation we check that any section the LLM cited
    actually appears in the retrieved sources. A cited-but-unretrieved section is a
    hallucination signal; we downgrade confidence and escalate rather than vouch for it.

Preserved from before: the Mode A/B prompt and deterministic confidence (the LLM's own
confidence is ignored — it is unreliable at self-assessment).
"""
from __future__ import annotations

import re
import time

from app.config import settings
from app.rag.models import Citation, RetrievedChunk
from app.rag.retriever import HybridRetriever
from app.schemas.ask import (
    DISCLAIMER,
    LEGAL_AID_ESCALATION,
    AskResponse,
    Confidence,
)
from app.services.llm_service import OllamaClient, get_llm
from app.utils.logger import get_logger

logger = get_logger(__name__)

_MODE_A_PREFIXES: tuple[str, ...] = ("used ",)
_MODE_B_PREFIXES: tuple[str, ...] = ("no strong context", "no context", "general principles")
_UNCERTAINTY_PHRASES: tuple[str, ...] = (
    "unclear",
    "uncertain",
    "not sure",
    "ambiguous",
    "without more information",
    "without additional context",
    "cannot be determined",
    "consult a lawyer to determine",
)

_CONTEXT_CHUNKS = 6
_MAX_CITATIONS = 5
# Cap how many current-law successor chunks we inject, so expansion enriches context
# without crowding out the directly-retrieved sources.
_MAX_EXPANSION = 4
_SECTION_WORD = r"(?:section|sec\.?|s\.?)"

# Extracts the base section/article number from a reference string:
# "IPC Section 420" -> "420", "BNS 318(4)" -> "318", "Article 21" -> "21".
_NUM_RE = re.compile(r"(\d+[A-Za-z]?)")
# Generic law_reference values that name no specific section to verify.
_GENERIC_REFS = ("general legal guidance", "general indian law", "general")


PROMPT_TEMPLATE = """You are an AI legal assistant for Indian law.

---
CONTEXT (each item is a retrieved source you may cite by its [LABEL]):
{context}

---
QUESTION:
{query}

---
DECISION LOGIC (CRITICAL — read carefully before answering):

MODE A — STRICT (use when context covers the question):
Trigger when the CONTEXT contains either:
  - a specific law section (BNS / IPC / CrPC / BNSS / Constitutional Article / IT Act) that matches the question, OR
  - a clear legal answer to the question.
In Mode A you MUST:
  - Use ONLY the context to answer.
  - Quote section numbers, punishments, and provisions EXACTLY as written in the context.
  - Prefer CURRENT law: if a current-code section (BNS/BNSS/BSA) and the repealed section it
    replaced (IPC/CrPC/IEA) are both present, LEAD with the current one and note the repealed
    code applies only to matters before 1 July 2024.
  - NOT paraphrase punishments. NOT generalize. NOT add your own knowledge.
  - Set "law_reference" to the exact section/article from context, copying the code name
    VERBATIM from its [LABEL] (e.g. "BNS Section 318", "BNSS Section 173", "BSA Section 23", "IPC Section 420").
  - Begin "reasoning" with "Used <label> from context".

MODE B — GENERAL GUIDANCE (use only when Mode A does not apply):
Trigger when the CONTEXT is weak, off-topic, or missing for this question.
In Mode B you MUST:
  - Provide helpful general guidance based on Indian legal principles.
  - Begin "answer" with "Typically under Indian law,".
  - Set "law_reference" to a broad reference (e.g. "Article 21" or "General Legal Guidance").
  - NOT invent specific section numbers or punishments. If unsure, omit the number.
  - Begin "reasoning" with "No strong context, used general principles".

NEVER mix Mode A and Mode B in the same answer.

---
SPECIAL RULE — PUNISHMENTS:
If the context names a section together with its punishment, quote that punishment
string verbatim. Do NOT write "punishment varies" when the context states a term.

SPECIAL RULE — CASE LAW (court judgments):
A context item labelled as a court judgment (a case name, or a "S.C.R." / "SCC" citation)
is ONE court's ruling on its own facts — persuasive, NOT the binding text of the law.
  - If a statute section in the context also answers the question, LEAD with the statute
    and set "law_reference" to that section, not the case.
  - Cite a judgment only to show how the law was applied, and write "the Court held …"
    rather than stating it as the law itself.
  - Do NOT put a case citation in "law_reference" when a statute is available.

SPECIAL RULE — CODE NAMES (do not confuse the three reform codes):
India's 2023 reforms created THREE SEPARATE codes that replaced three old ones:
  - BNS  (Bharatiya Nyaya Sanhita)            — the PENAL code; replaced the IPC.
  - BNSS (Bharatiya Nagarik Suraksha Sanhita) — the PROCEDURE code; replaced the CrPC.
  - BSA  (Bharatiya Sakshya Adhiniyam)        — the EVIDENCE act; replaced the Evidence Act.
They are DIFFERENT codes with their own section numbers. If the context [LABEL] says
"BNSS Section 173", the answer is BNSS 173 — NOT "BNS 173". Copy the code name exactly as
written in the [LABEL]; never shorten BNSS or BSA to "BNS", and never relabel one code as another.

SAFETY:
- Do NOT cite any section number that is not present in the CONTEXT above.
- Do NOT invent punishments. If unsure, lower the confidence.

STYLE:
- Simple language for an Indian audience. One concrete actionable next step.

---
Return ONLY this JSON object (no prose before or after):
{{
  "answer": "...clear explanation...",
  "law_reference": "...exact code+section from a [LABEL] e.g. BNS/BNSS/BSA/IPC/CrPC/Article, or 'General Legal Guidance'...",
  "action": "...what user should do next...",
  "confidence": "high|medium|low",
  "reasoning": "brief: 'Used BNS 318 from context' OR 'No strong context, used general principles'"
}}
"""


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def _enforce_confidence(reasoning: str, answer: str) -> Confidence:
    r = reasoning.strip().lower()
    a = answer.lower()
    if any(r.startswith(p) for p in _MODE_A_PREFIXES):
        return "high"
    if any(r.startswith(p) for p in _MODE_B_PREFIXES):
        if any(phrase in a for phrase in _UNCERTAINTY_PHRASES):
            return "low"
        return "medium"
    return "medium"


def _stringify(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v).strip() for v in value if v)
    return str(value).strip()


def _base_num(token: str) -> str:
    m = _NUM_RE.search(token or "")
    return m.group(1) if m else ""


def _format_context(results: list[RetrievedChunk]) -> str:
    lines: list[str] = []
    for rc in results:
        c = rc.chunk
        label = c.reference_label()
        if c.code_status == "repealed":
            tag = " (REPEALED — applies only to offences before 1 July 2024)"
        elif c.code_status == "current":
            tag = " (CURRENT LAW)"
        else:
            tag = ""
        lines.append(f"[{label}{tag}] {c.text}")
    return "\n\n".join(lines)


def _order_for_context(results: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Stable reorder so CURRENT statutes lead, then repealed, then everything else —
    without disturbing the reranker's ordering within each group."""
    def rank(rc: RetrievedChunk) -> int:
        c = rc.chunk
        if c.source_type == "statute" and c.code_status == "current":
            return 0
        if c.source_type == "statute" and c.code_status == "repealed":
            return 1
        return 2

    return sorted(results, key=rank)


def _verify_citation(law_reference: str, results: list[RetrievedChunk]) -> bool:
    """True unless the answer cited a specific section that no retrieved source contains.

    Lenient by design (avoid false alarms): generic/constitutional references pass, and
    if *any* cited section is present we accept it. Only a citation with no support at
    all trips the flag."""
    ref = law_reference.strip().lower()
    if not ref or any(g in ref for g in _GENERIC_REFS) or "article" in ref:
        return True

    # Normalize section tokens to a single case: a cited "124A" must match a retrieved
    # "124A" even though ``ref`` was lower-cased above. Extract from the original
    # reference and upper-case both sides so letter-suffix sections (124A, 304B, 326A,
    # 376D, …) aren't falsely flagged unverified.
    cited = {_base_num(t).upper() for t in _NUM_RE.findall(law_reference)}
    cited.discard("")
    if not cited:
        return True

    available: set[str] = set()
    for rc in results:
        if rc.chunk.section:
            available.add(_base_num(rc.chunk.section).upper())
        if rc.chunk.article:
            available.add(_base_num(rc.chunk.article).upper())

    return bool(cited & available)


def _scan_repealed_refs(text: str, from_codes: list[str]) -> list[tuple[str, str]]:
    """Ordered, de-duped (code, section) repealed-law references found in free text.

    Finds "(IPC|CrPC|IEA) Section N" style mentions in a query or a plain-language guide,
    so the engine can pull the CURRENT successor into context even when the repealed
    section isn't itself a retrieved, section-tagged statute chunk (e.g. an FIR guide
    that only narrates "Section 154 CrPC")."""
    if not text:
        return []
    hits: list[tuple[int, tuple[str, str]]] = []
    for code in from_codes:
        c = re.escape(code)
        # CODE [section] N — "IPC 420", "CrPC Section 154", "IPC s. 124A"
        for m in re.finditer(rf"\b{c}\b[\s.,]*{_SECTION_WORD}?\s*(\d+[A-Za-z]?)", text, re.IGNORECASE):
            hits.append((m.start(), (code, m.group(1))))
        # section N(..) CODE — "Section 154(3) CrPC"
        for m in re.finditer(rf"{_SECTION_WORD}\s*(\d+[A-Za-z]?)\s*\(?\d*\)?\s*\b{c}\b", text, re.IGNORECASE):
            hits.append((m.start(), (code, m.group(1))))
        # section N of [the] CODE — "section 420 of the IPC"
        for m in re.finditer(rf"{_SECTION_WORD}\s*(\d+[A-Za-z]?)[^.\n]*?\bof\s+(?:the\s+)?\b{c}\b", text, re.IGNORECASE):
            hits.append((m.start(), (code, m.group(1))))
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for _, ref in sorted(hits, key=lambda h: h[0]):
        if ref not in seen:
            seen.add(ref)
            out.append(ref)
    return out


def _current_ref_in_note(note: str, to_codes: list[str]) -> tuple[str, str] | None:
    """Pull a current-code section named in a mapping note, e.g. 'BNS s.152' -> ('BNS','152').

    Used for repealed sections with no 1:1 successor (like sedition, IPC 124A) whose curated
    note points at the new, differently-framed provision often discussed in its place."""
    if not note:
        return None
    for code in sorted(to_codes, key=len, reverse=True):  # try BNSS before BNS
        c = re.escape(code)
        m = re.search(rf"\b{c}\b[\s.]*(?:section|s\.?)?\s*(\d+[A-Za-z]?)", note, re.IGNORECASE)
        if m:
            return (code, m.group(1))
    return None


# --------------------------------------------------------------------------- #
class RAGService:
    def __init__(
        self,
        llm: OllamaClient | None = None,
        retriever: HybridRetriever | None = None,
    ) -> None:
        self._llm = llm or get_llm()
        self._retriever = retriever or HybridRetriever()
        self._abstain_threshold = float(getattr(settings, "min_rerank_score", -10.0))

    # ------------------------------------------------------------------ #
    def answer(self, query: str, language: str = "en") -> AskResponse:
        started = time.perf_counter()
        query = query.strip()

        results = self._retriever.retrieve(query)
        top_score = results[0].score if results else None
        logger.info(
            "Retrieved %d chunks (top_score=%s) for query=%r",
            len(results),
            f"{top_score:.3f}" if top_score is not None else "none",
            query[:120],
        )

        if not results or (top_score is not None and top_score < self._abstain_threshold):
            return self._abstain(query, results, started)

        # Phase 2: ensure current law (BNS/BNSS/BSA) is in context for any repealed
        # reference in the question or the retrieved sources, so we can lead with it.
        results = self._expand_current_law(results, query)
        ordered = _order_for_context(results)

        context = _format_context(ordered[:_CONTEXT_CHUNKS])
        prompt = PROMPT_TEMPLATE.format(context=context, query=query)
        raw = self._llm.generate_json(prompt)

        answer = _stringify(raw.get("answer")) or (
            "Typically under Indian law, this question needs a closer look at the facts and provisions."
        )
        law_reference = _stringify(raw.get("law_reference")) or "General Indian law"
        action = _stringify(raw.get("action")) or (
            "Consult a qualified lawyer for guidance specific to your situation."
        )
        reasoning = _stringify(raw.get("reasoning")) or "No strong context, used general principles."

        llm_confidence = _stringify(raw.get("confidence")).lower() or "?"
        confidence = _enforce_confidence(reasoning, answer)

        # Deterministic current-law correction: the small model often headlines the
        # repealed section it read (e.g. CrPC 154 from an FIR guide) instead of the
        # successor in force. If the cited repealed section has a verified 1:1 successor
        # present in context, rewrite the headline citation to that current section — the
        # engine's promise (lead with law in force), enforced not left to the LLM.
        law_reference = self._prefer_current_reference(law_reference, ordered)

        # Phase 2: hallucination gate — did the LLM cite a section we actually retrieved?
        citation_verified = _verify_citation(law_reference, ordered)
        if not citation_verified:
            logger.warning("Citation unverified: %r not in retrieved sources — downgrading.", law_reference)
            confidence = "low"

        current_law_note = self._current_law_note(ordered, query)
        citations = self._dedupe_citations(ordered)
        escalation = (
            LEGAL_AID_ESCALATION if (confidence == "low" or not citation_verified) else None
        )

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "Answered in %dms | confidence=%s (llm_said=%s) | verified=%s",
            elapsed_ms,
            confidence,
            llm_confidence,
            citation_verified,
        )

        return AskResponse(
            answer=answer,
            law_reference=law_reference,
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            citations=citations,
            abstained=False,
            escalation=escalation,
            current_law_note=current_law_note,
            citation_verified=citation_verified,
            disclaimer=DISCLAIMER,
            response_time_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------ #
    # Phase 2 helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _collect_repealed_refs(
        results: list[RetrievedChunk], query: str, from_codes: list[str]
    ) -> list[tuple[str, str]]:
        """Ordered repealed (code, base-section) references to bridge to current law —
        gathered from the QUESTION, from retrieved repealed statutes (by metadata), and
        from the text of retrieved guides/QA (so a CrPC-only guide still routes to BNSS).
        Query refs lead, then refs in retrieval order."""
        fc = set(from_codes)
        refs: list[tuple[str, str]] = []

        def add(items: list[tuple[str, str]]) -> None:
            for code, sec in items:
                ref = (code, _base_num(sec))
                if ref[1] and ref not in refs:
                    refs.append(ref)

        add(_scan_repealed_refs(query, from_codes))
        for rc in results:
            c = rc.chunk
            if c.act in fc and c.section:           # retrieved repealed statute (trust its metadata)
                add([(c.act, c.section)])
            else:                                    # guide / QA / judgment — scan the prose
                add(_scan_repealed_refs(f"{c.text} {c.title or ''}", from_codes))
        return refs

    def _expand_current_law(self, results: list[RetrievedChunk], query: str) -> list[RetrievedChunk]:
        """Pull the current successor chunk (BNS/BNSS/BSA) for any repealed section
        (IPC/CrPC/IEA) referenced by the question or the retrieved sources, so the answer
        can LEAD with — and verifiably cite — law in force. For a repealed section with no
        1:1 successor (e.g. sedition, IPC 124A) we fall back to the current provision named
        in the mapping note (BNS 152). Best-effort: a failure leaves results unchanged."""
        try:
            from app.rag.law_map import LawMap
            from app.rag.vector_store import VectorStore

            law_map = LawMap.instance()
            from_codes = list(law_map.from_codes())
            to_codes = list(law_map.to_codes())
            refs = self._collect_repealed_refs(results, query, from_codes)
            if not refs:
                return results

            present = {(rc.chunk.act, _base_num(rc.chunk.section or "")) for rc in results}
            wanted: dict[str, set[str]] = {}  # current code -> base sections to pull
            for code, sec in refs:
                entry = law_map.successor(code, sec)
                if not entry:
                    continue
                if entry.get("new"):
                    # Fetch by BASE section ("173"); the bare-act index is keyed by base,
                    # not the mapping's subsection token ("173(1)(ii)").
                    target = (entry["to_code"], _base_num(entry["new"]))
                else:
                    nb = _current_ref_in_note(entry.get("note", ""), to_codes)
                    if not nb:
                        continue
                    target = (nb[0], _base_num(nb[1]))
                if target[1] and target not in present:
                    wanted.setdefault(target[0], set()).add(target[1])

            added = 0
            for to_code, sections in wanted.items():
                if added >= _MAX_EXPANSION:
                    break
                extra = VectorStore.instance().fetch_by_reference(act=to_code, sections=sorted(sections))
                for chunk in extra:
                    if added >= _MAX_EXPANSION:
                        break
                    results.append(RetrievedChunk(chunk=chunk, score=0.0))
                    added += 1
            if added:
                logger.info("Cross-ref expansion added %d current-law chunk(s) from %d ref(s)", added, len(refs))
        except Exception as e:  # never let expansion break a query
            logger.warning("Current-law expansion skipped: %s", e)
        return results

    @staticmethod
    def _prefer_current_reference(law_reference: str, ordered: list[RetrievedChunk]) -> str:
        """Rewrite a repealed headline citation to its current successor when that
        successor is present in context (so it's verifiable). Only acts on a genuine 1:1
        successor — a repealed section with no direct equivalent (e.g. sedition, IPC 124A)
        is left as-is for the gate to flag, since claiming a successor would mislead."""
        from app.rag.law_map import LawMap

        law_map = LawMap.instance()
        refs = _scan_repealed_refs(law_reference, list(law_map.from_codes()))
        if not refs:
            return law_reference
        code, sec = refs[0]
        entry = law_map.successor(code, sec)
        if not (entry and entry.get("new")):
            return law_reference
        to_code, new_base = entry["to_code"], _base_num(entry["new"])
        present = {(rc.chunk.act, _base_num(rc.chunk.section or "")) for rc in ordered}
        if new_base and (to_code, new_base) in present:
            new_label = f"{to_code} Section {new_base}"
            logger.info("Current-law correction: headline %r -> %r", law_reference, new_label)
            return new_label
        return law_reference

    def _current_law_note(self, results: list[RetrievedChunk], query: str) -> str | None:
        """Bridge the first repealed section referenced (by the question or the retrieved
        sources) to its current successor, with a caveat when the map is unverified."""
        from app.rag.law_map import LawMap

        law_map = LawMap.instance()
        for code, sec in self._collect_repealed_refs(results, query, list(law_map.from_codes())):
            note = law_map.current_reference_note(code, sec)
            if note:
                if not law_map.verified_for(code):
                    note += " (Mapping is indicative — confirm against the official bare act.)"
                return note
        return None

    @staticmethod
    def _dedupe_citations(results: list[RetrievedChunk]) -> list[Citation]:
        """First-N unique citations by label, preserving (current-first) order."""
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

    # ------------------------------------------------------------------ #
    def _abstain(self, query: str, results: list[RetrievedChunk], started: float) -> AskResponse:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info("Abstained (weak retrieval) in %dms for query=%r", elapsed_ms, query[:120])
        return AskResponse(
            answer=(
                "I couldn't find a reliable basis in my legal sources to answer this "
                "confidently. To avoid giving you wrong legal information, I'd rather not "
                "guess. Please rephrase with more detail, or seek the help below."
            ),
            law_reference="General Legal Guidance",
            action="Contact free legal aid or a qualified lawyer for your specific situation.",
            confidence="low",
            reasoning="No strong context, used general principles.",
            # Abstaining means nothing scored as reliably relevant — don't dangle the
            # weak/below-threshold chunks as if they were sources for an answer.
            citations=[],
            abstained=True,
            escalation=LEGAL_AID_ESCALATION,
            current_law_note=None,
            citation_verified=True,
            disclaimer=DISCLAIMER,
            response_time_ms=elapsed_ms,
        )
