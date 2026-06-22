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
    DISCLAIMER_HI,
    LEGAL_AID_ESCALATION,
    LEGAL_AID_ESCALATION_HI,
    AskResponse,
    CaseAnalysis,
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

# --- Hindi (multilingual) ---
# A citizen who is more comfortable in Hindi should be able to ask in Hindi and get a
# trustworthy Hindi answer. We keep the strong English legal *retrieval* (translate the
# query to English first) and ask the LLM to *answer* in Hindi while keeping every law
# reference precise and standard — so the trust contract (citations, current-law,
# abstention) is unchanged; only the prose language changes.
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")

_HINDI_INSTRUCTION = """
---
LANGUAGE — ANSWER IN HINDI:
Write "answer" and "action" in simple, clear Hindi (Devanagari) an ordinary person understands.
Keep every law reference precise and recognizable: code names in standard form (BNS, BNSS, BSA, IPC, CrPC, Article)
and section numbers as digits (you may use "धारा" for "Section"). Set "law_reference" to the standard English form
(e.g. "BNS Section 103"). Keep "reasoning" in English (begin "Used ..." or "No strong context ...").
Also write the analysis arrays (situation, what_happens_next, do_now, also_possible, for_your_advocate) in simple Hindi.
In applicable_law keep code names (BNS/BNSS/BSA/IPC/CrPC) and section numbers in standard English/digits.
For also_possible, begin each item with "कानून अनुमति देता है…", "अदालत विचार कर सकती है…", या "यह अदालत तय करेगी कि…".
"""

_ABSTAIN_ANSWER_HI = (
    "मुझे अपने कानूनी स्रोतों में इसका भरोसेमंद उत्तर देने का पर्याप्त आधार नहीं मिला। "
    "ग़लत कानूनी जानकारी देने से बचने के लिए मैं अनुमान नहीं लगाना चाहूँगा। कृपया अधिक "
    "विवरण के साथ दोबारा पूछें, या नीचे दी गई सहायता लें।"
)
_ABSTAIN_ACTION_HI = "अपनी स्थिति के लिए मुफ़्त कानूनी सहायता या किसी योग्य वकील से संपर्क करें।"
_FALLBACK_ANSWER_HI = "आमतौर पर भारतीय कानून के अनुसार, इस प्रश्न के लिए तथ्यों और प्रावधानों को बारीकी से देखना होगा।"
_FALLBACK_ACTION_HI = "अपनी स्थिति के अनुसार मार्गदर्शन के लिए किसी योग्य वकील से सलाह लें।"


def _is_hindi(language: str, query: str) -> bool:
    """Answer in Hindi if the UI asked for it OR the question itself is in Devanagari."""
    return (language or "").strip().lower().startswith("hi") or bool(_DEVANAGARI.search(query or ""))


# A lay person describes a situation in the first/third person ("the shopkeeper sold ME a
# defective mixer", "I bought a fridge and it arrived broken", "they won't refund my
# money"). Such prose names no statute and reranks deeply negative verbatim, so it needs
# the rewrite even when it's a touch under the 12-word narrative bar. Pure keyword lookups
# ("grievous hurt punishment", "knife attack ICU") carry none of these markers.
_NARRATIVE_RE = re.compile(
    r"\b(?:i|we|me|my|us|our|he|she|they|him|her|them|his|their)\b"
    r"|\bthe\s+\w+\s+(?:sold|gave|took|charged|refus|deni|damag|lost|broke|cheat|deliver|"
    r"replac|return|paid|promis|sent|kept|won)",
    re.IGNORECASE,
)


def _is_lay_narrative(query: str) -> bool:
    """True if the text reads like a described situation (first/third-person storytelling)
    rather than a keyword lookup — the case where verbatim retrieval fails and a distilling
    rewrite is needed."""
    return bool(_NARRATIVE_RE.search(query or ""))


# Cap how many current-law successor chunks we inject, so expansion enriches context
# without crowding out the directly-retrieved sources.
_MAX_EXPANSION = 4
_SECTION_WORD = r"(?:section|sec\.?|s\.?)"

# Extracts the base section/article number from a reference string:
# "IPC Section 420" -> "420", "BNS 318(4)" -> "318", "Article 21" -> "21".
_NUM_RE = re.compile(r"(\d+[A-Za-z]?)")
# Generic law_reference values that name no specific section to verify.
_GENERIC_REFS = ("general legal guidance", "general indian law", "general")
# Death-deeming / aggravated offences whose grounded classification must never be the
# auto-headline: their applicability turns on a fact (a death) the section number alone
# doesn't establish, so surfacing e.g. BNS 80 (Dowry Death, Court of Session) for a LIVING
# dowry-cruelty victim over-states both gravity and trial forum. Suppress over mislabel.
_NO_AUTO_CLASS = {"80"}

# --------------------------------------------------------------------------- #
# Case-analysis (situation → structured guidance) safety machinery.
# The rich "analysis" block is built ONLY on a strong, citation-verified Mode-A answer
# (see RAGService._build_analysis). Everything below is defence-in-depth *behind* the
# prompt: deterministic scrubbers that keep every bullet inside the trust contract, so a
# prompt slip can never surface an outcome prediction, an ungrounded offence label, or an
# invented precedent to a citizen or their lawyer.
# --------------------------------------------------------------------------- #
_ANALYSIS_KEYS = (
    "situation", "applicable_law", "what_happens_next", "do_now", "also_possible", "for_your_advocate",
)
_ANALYSIS_MAX_ITEMS = 8
_ANALYSIS_MAX_LEN = 400

# The mandatory frame rendered atop any analysis block — set in code, never from the LLM.
_OUTCOME_FRAMING = (
    "This explains what the law provides and what a court examines — not a prediction of your case."
)
_OUTCOME_FRAMING_HI = (
    "यह बताता है कि कानून क्या प्रावधान देता है और अदालत किन बातों को देखती है — "
    "यह आपके मामले के परिणाम की भविष्यवाणी नहीं है।"
)

# Offence-classification labels are NOT in our sources (the BNSS First Schedule isn't
# ingested), so any such label in synthesised prose is ungrounded — strip the bullet.
# Includes Hindi forms so a Devanagari bullet isn't silently exempt.
_CLASSIFICATION_RE = re.compile(
    r"\b(?:non[-\s]?)?cognizable\b|\b(?:non[-\s]?)?bailable\b|\b(?:non[-\s]?)?compoundable\b|\btriable\b"
    r"|संज्ञेय|असंज्ञेय|ज़मानती|जमानती|गैर[-\s]?ज़मानती|गैर[-\s]?जमानती|शमनीय|असंज्ञेय",
    re.IGNORECASE,
)

# Second-person / likelihood verdict phrasing — a de-facto outcome prediction. Dropped
# wherever it appears (English and Hindi). Backstops the safe-stem allowlist below.
_OUTCOME_PREDICTION_RE = re.compile(
    r"\byou(?:'ll| will| are going to| can expect| should (?:get|win|receive)| have a good chance| are likely)\b"
    r"|\byour (?:sentence|case is (?:strong|weak)|chances?)\b"
    r"|\bgood chance\b|\bguaranteed\b|\blikely to (?:get|be|win|lose|receive)\b"
    r"|\b(?:the )?accused will (?:be|get)\b"
    r"|आपको\s+\S+\s*(?:मिलेगी|मिलेगा|मिल\s*जाएगी|मिल\s*जाएगा|होगी|होगा)"
    r"|आप\s+\S*\s*(?:जाएंगे|जाएगा|बरी)"
    r"|आपकी\s*सज़ा|आपका\s*मामला\s*(?:मजबूत|मज़बूत|कमज़ोर)|संभावना\s*है\s*कि\s*आप",
    re.IGNORECASE,
)

# Case-citation shaped tokens. We have NO offence→precedent map, so no case may EVER be
# named in advocate notes; a bullet that looks like it cites one is dropped.
_PRECEDENT_RE = re.compile(
    r"\bv\.?\s+[A-Z]\w|\bvs\.?\s+[A-Z]\w|\bSCC\b|\bS\.?\s?C\.?\s?R\.?\b|\bAIR\b|\b\d{4}\s*SCC\b",
)

# --------------------------------------------------------------------------- #
# Reporter-citation scrub (fabrication-adjacent risk the audit flagged once).
# A precise LAW-REPORT citation string — "(2024) 10 SCC 1", "[2024] 1 S.C.R. 1134",
# "AIR 2024 SC 567" — appended by the LLM but NOT present verbatim in any retrieved
# source chunk is an invented precedent reference. Case NAMES from the corpus are fine;
# only the precise reporter token (or its enclosing parenthetical) is at risk. We strip
# such a token from the user-facing 'answer'/'reasoning' UNLESS it appears verbatim in a
# visible source chunk's text. Deterministic, surgical, never raises.
#
# Each pattern captures the full reporter token, e.g.:
#   SCC  : "(2024) 10 SCC 1"        and bare "2024 SCC 1" / "(2024) 10 S.C.C. 1"
#   SCR  : "[2024] 1 S.C.R. 1134"   (square or round brackets, dotted or not)
#   AIR  : "AIR 2024 SC 567"        (AIR <year> <court> <num>)
# Case-insensitive, tolerant of internal spacing (incl. dotted "S.C.C." / "S.C.R.").
_REPORTER_CITATION_RES: tuple[re.Pattern[str], ...] = (
    # SCR — bracketed year then "S.C.R." / "SCR": "[2024] 1 S.C.R. 1134", "(2024) SCR 5"
    re.compile(
        r"[\[(]\s*\d{4}\s*[\])]\s*(?:\d+\s+)?S\.?\s?C\.?\s?R\.?\s*\d+",
        re.IGNORECASE,
    ),
    # SCC — bracketed year then "SCC": "(2024) 10 SCC 1", "[2024] 10 S.C.C. 1"
    re.compile(
        r"[\[(]\s*\d{4}\s*[\])]\s*(?:\d+\s+)?S\.?\s?C\.?\s?C\.?\s*\d+",
        re.IGNORECASE,
    ),
    # AIR — "AIR 2024 SC 567" (court abbrev. 2-4 letters: SC, Del, Bom, All, …)
    re.compile(
        r"\bA\.?\s?I\.?\s?R\.?\s*\d{4}\s*[A-Za-z]{2,4}\s*\d+",
        re.IGNORECASE,
    ),
    # Bare SCC/SCR without brackets: "2024 SCC 1", "2024 10 SCR 5"
    re.compile(
        r"\b\d{4}\s+(?:\d+\s+)?S\.?\s?C\.?\s?[CR]\.?\s*\d+",
        re.IGNORECASE,
    ),
)

# Collapse any run of whitespace so a citation that the LLM spaced differently than the
# source ("S.C.R.  1134" vs "S.C.R. 1134") still matches verbatim-ish against the chunk.
_WS_RE = re.compile(r"\s+")


def _norm_citation(s: str) -> str:
    """Lower-cased, whitespace-collapsed form for verbatim-tolerant presence checks."""
    return _WS_RE.sub(" ", (s or "")).strip().lower()


def _tidy_after_scrub(text: str) -> str:
    """Repair punctuation left dangling after a citation token is excised, so the
    sentence stays readable. Drops empty () / [] husks, doubled spaces, and a space
    before sentence punctuation; never touches sentence content."""
    # Empty parenthetical husks left behind, e.g. "the Court held (  )." -> "the Court held."
    text = re.sub(r"\(\s*[,;]?\s*\)", "", text)
    text = re.sub(r"\[\s*[,;]?\s*\]", "", text)
    # "see ( , )" style leftovers handled above; now collapse spacing.
    text = _WS_RE.sub(" ", text)
    # Space before terminal/clausal punctuation, and doubled punctuation.
    text = re.sub(r"\s+([,.;:)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\s*,\s*\.", ".", text)
    return text.strip()


def _scrub_reporter_citations(text: str, sources: list[RetrievedChunk]) -> str:
    """Remove precise law-report citation strings (SCC / SCR / AIR) NOT present verbatim
    in any retrieved source chunk, leaving the surrounding sentence readable.

    A citation token is KEPT only if its whitespace-collapsed, case-folded form is a
    substring of some visible source chunk's text — i.e. the corpus actually contains it.
    Otherwise the token is excised (along with an immediately-enclosing parenthetical when
    that parenthetical held only the citation) and the residue tidied. Defensive: any
    failure returns the input unchanged (we never raise, and never weaken the answer)."""
    try:
        if not text:
            return text
        haystack = " ".join(_norm_citation(rc.chunk.text) for rc in (sources or []))

        def is_grounded(token: str) -> bool:
            return _norm_citation(token) in haystack

        # Collect (start, end) spans to remove, widening to swallow a parenthetical that
        # wrapped ONLY the citation (e.g. " (AIR 2024 SC 567)" -> "").
        spans: list[tuple[int, int]] = []
        for rx in _REPORTER_CITATION_RES:
            for m in rx.finditer(text):
                if is_grounded(m.group(0)):
                    continue
                start, end = m.start(), m.end()
                # If the citation sits inside a parenthetical that contains nothing else of
                # substance, remove the whole "(...)" including a leading space.
                lp = text.rfind("(", 0, start)
                rp = text.find(")", end)
                if lp != -1 and rp != -1:
                    inner = text[lp + 1:rp]
                    # only the citation (+ light connective words) lived in the parens
                    residue = inner[: start - lp - 1] + inner[end - lp - 1:]
                    if re.fullmatch(r"[\s,;:]*(?:see|cf\.?|e\.g\.?|citing)?[\s,;:]*", residue, re.IGNORECASE):
                        start = lp
                        if start > 0 and text[start - 1] == " ":
                            start -= 1
                        end = rp + 1
                spans.append((start, end))

        if not spans:
            return text

        # Merge overlapping spans, then excise from the end so indices stay valid.
        spans.sort()
        merged: list[tuple[int, int]] = []
        for s, e in spans:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        out = text
        for s, e in reversed(merged):
            out = out[:s] + out[e:]
        return _tidy_after_scrub(out)
    except Exception as exc:  # never let scrubbing break a query
        logger.warning("Reporter-citation scrub skipped: %s", exc)
        return text


# "also_possible" carries the highest outcome-prediction risk, so it is allowlisted: a
# bullet survives only if it OPENS with an approved impersonal stem (EN or HI). Anything
# that isn't framed as "the law / a court may…" is dropped rather than trusted.
_SAFE_STEMS = (
    "the law allows", "the law provides", "the law permits", "a court may", "a court can",
    "a court decides", "a court would", "courts may", "whether ", "it may be possible",
    "depending on the facts", "in some cases", "the code allows", "the statute provides",
    "कानून अनुमति", "कानून के अनुसार", "कानून यह", "अदालत", "यह अदालत", "तथ्यों के आधार",
    "कुछ मामलों", "हो सकता है", "संहिता",
)


def _available_sections(chunks: list[RetrievedChunk]) -> set[str]:
    """Upper-cased base section/article numbers present in the given chunks (the same
    notion of 'available' the citation gate uses), for per-bullet grounding checks."""
    av: set[str] = set()
    for rc in chunks:
        if rc.chunk.section:
            av.add(_base_num(rc.chunk.section).upper())
        if rc.chunk.article:
            av.add(_base_num(rc.chunk.article).upper())
    av.discard("")
    return av


def _bnss_sections(chunks: list[RetrievedChunk]) -> set[str]:
    """Base section numbers that belong specifically to BNSS chunks — so 'what happens
    next' procedure steps can only be grounded in the actual procedure code."""
    av: set[str] = set()
    for rc in chunks:
        if (rc.chunk.act or "").upper() == "BNSS" and rc.chunk.section:
            av.add(_base_num(rc.chunk.section).upper())
    av.discard("")
    return av


# The canonical BNSS criminal-process arc. An offence query ("he took my money")
# retrieves the OFFENCE (BNS) but never the PROCEDURE (BNSS), so "what happens next" has
# nothing to ground on. For a criminal matter we deterministically inject these real,
# verified BNSS sections into context so the model can build the process steps from
# statute — FIR → investigate → chargesheet → cognizance → charge → judgment.
_BNSS_PROCESS_ARC: tuple[str, ...] = ("173", "175", "176", "193", "210", "251", "258")


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

SPECIAL RULE — CASE ANALYSIS (the "analysis" object):
Fill "analysis" ONLY in Mode A (strong, on-point context). In Mode B, set EVERY analysis array to [].
Every item must be supported by the CONTEXT above; if you have no support for an array, return [] for it.
  - situation: 1-3 plain sentences naming what the facts legally are (e.g. "This appears to involve cheating").
    NO section numbers here. Never write "you committed"; use "This appears to involve…".
  - applicable_law: one item per section, copied from a [LABEL], CURRENT (BNS/BNSS/BSA) first, with the
    punishment exactly as written; mention any old section only as a parenthetical "(was IPC 420)".
    Only sections that appear in the CONTEXT.
  - what_happens_next: ordered procedure steps — include a step ONLY if its BNSS section is in the CONTEXT,
    and name that BNSS section in the step. If no BNSS procedure section is in the CONTEXT, return [].
  - do_now: 2-4 calm, concrete actions an ordinary person can take now. No section numbers, no promises
    about the result.
  - also_possible: options such as bail / settlement / compounding / civil remedy / a defence — each written
    as an IMPERSONAL possibility beginning "The law allows…", "A court may…", or "Whether … is for the court
    to decide". NEVER "you will…", "you are likely…", "you should get…", or "your case is strong".
  - for_your_advocate: ingredients to prove (from the cited section), limitation, jurisdiction, and what to
    research. Do NOT name any court case, party name, or citation — write "leading judgments on this offence".
  - OUTCOMES: describe only what the STATUTE PROVIDES and what a COURT EXAMINES. NEVER predict this person's
    result. Forbidden: "you will get", "you will be convicted/acquitted", "your sentence will be", "guaranteed".
  - CLASSIFICATION: do NOT state whether an offence is cognizable / non-cognizable, bailable / non-bailable,
    compoundable, or which court tries it — that information is not in your sources.

STYLE:
- Simple language for an Indian audience. One concrete actionable next step.

---
Return ONLY this JSON object (no prose before or after):
{{
  "answer": "...clear explanation...",
  "law_reference": "...exact code+section from a [LABEL] e.g. BNS/BNSS/BSA/IPC/CrPC/Article, or 'General Legal Guidance'...",
  "action": "...what user should do next...",
  "confidence": "high|medium|low",
  "reasoning": "brief: 'Used BNS 318 from context' OR 'No strong context, used general principles'",
  "analysis": {{
    "situation": ["plain sentence(s) on what the facts legally are — NO section numbers"],
    "applicable_law": ["one bullet per section from a [LABEL], CURRENT first, with punishment; old section only as '(was IPC 420)'"],
    "what_happens_next": ["ordered steps — ONLY if a BNSS section in CONTEXT supports the step, naming it; ELSE []"],
    "do_now": ["2-4 concrete, calm steps — no section numbers, no result promises"],
    "also_possible": ["bail / settlement / civil option / defence — begin 'The law allows…' or 'A court may…'; NEVER 'you will…'; [] if unsupported"],
    "for_your_advocate": ["ingredients to prove, limitation, jurisdiction; NO case names — say 'leading judgments on this offence'"]
  }}
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


# Pull a current-code (BNS/BNSS/BSA) section named in a free-text reference string, e.g.
# "BNS Section 307 (was IPC 307)" -> ("BNS", "307"). Returns the FIRST/lead current ref —
# the headline the answer leads with. Mirrors _current_ref_in_note but lead-anchored.
def _lead_current_ref(text: str, to_codes: list[str]) -> tuple[str, str] | None:
    if not text:
        return None
    best: tuple[int, str, str] | None = None
    for code in to_codes:
        c = re.escape(code)
        m = re.search(rf"\b{c}\b[\s.]*(?:section|sec\.?|s\.?|धारा)?\s*(\d+[A-Za-z]?)", text, re.IGNORECASE)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), code, m.group(1))
    return (best[1], best[2]) if best else None


def _current_law_guard_violation(law_reference: str) -> tuple[str, str, str] | None:
    """The grave-offence current-law guard: detect a headline citation whose CURRENT
    (BNS/BNSS/BSA) lead section is internally CONTRADICTED by a repealed section named in
    the SAME reference string, per the curated LawMap.

    The fragile, high-stakes case this protects against: an attempt-on-life fact pattern
    (victim alive + intent to kill = IPC 307 -> BNS 109) that the small model sometimes
    headlines as "BNS Section 307" — which is a *theft* offence (Theft after preparation…,
    per data/bns/bns_sections.csv, the successor of IPC 382, NOT IPC 307). When the model
    writes a headline like "BNS Section 307 (was IPC 307)", that is provably self-
    contradictory: LawMap says IPC 307's successor is BNS 109, not BNS 307. We detect ONLY
    that mapping contradiction — purely from the curated table, never from offence keywords —
    so the check can never mislabel a case whose framing happens to mention a weapon or the
    word "kill".

    Returns (current_code, current_section, repealed_ref) describing the contradiction when
    the SAME reference names a current lead section AND a repealed section whose verified 1:1
    successor is a DIFFERENT base section than that lead. Returns None otherwise (the common,
    consistent case — e.g. "BNS 307 (was IPC 382)" or "BNS 103 (was IPC 302)" both check
    out). Never raises."""
    try:
        from app.rag.law_map import LawMap

        law_map = LawMap.instance()
        to_codes = list(law_map.to_codes())
        lead = _lead_current_ref(law_reference, to_codes)
        if not lead:
            return None
        lead_code, lead_sec = lead[0], _base_num(lead[1])
        if not lead_sec:
            return None
        for code, sec in _scan_repealed_refs(law_reference, list(law_map.from_codes())):
            entry = law_map.successor(code, sec)
            if not (entry and entry.get("new")):
                continue
            # The repealed section maps to a DIFFERENT current code/section than the headline
            # claims — and the headline reuses the repealed section's OWN number on the wrong
            # code (the IPC 307 -> BNS 307 footgun). Both conditions = a provable contradiction.
            succ_code = (entry.get("to_code") or "").upper()
            succ_base = _base_num(entry["new"])
            if succ_code == lead_code.upper() and succ_base and succ_base != lead_sec:
                return (lead_code, lead_sec, f"{code} {sec}")
        return None
    except Exception as exc:  # never let the guard break a query
        logger.warning("Current-law guard skipped: %s", exc)
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
    def answer(self, query: str, language: str = "en", history: list | None = None) -> AskResponse:
        started = time.perf_counter()
        query = query.strip()
        history = list(history or [])
        hindi = _is_hindi(language, query)
        # Resolve the question into a standalone English search query — condense a follow-up
        # using recent history ("what's the punishment for that?") and/or translate Hindi —
        # so retrieval stays strong. History is used ONLY to disambiguate, never as a source.
        search_query = self._standalone_query(query, history, hindi)

        # Robust retrieval: the lay-narrative rewrite that rescues a poorly-retrieving
        # situation ("the shopkeeper sold me a defective mixer and won't refund") is
        # NONDETERMINISTIC — one sampling produces a strong statute-matching query, the next
        # a weak one that reranks below the abstain bar and would (wrongly) abstain. We make
        # the decision stable by, when the first rewrite lands below threshold, trying one
        # fresh rewrite AND the original text, then keeping the best-scoring merged set.
        results, search_query = self._robust_retrieve(query, search_query, history, hindi)
        top_score = results[0].score if results else None
        logger.info(
            "Retrieved %d chunks (top_score=%s) for query=%r (hindi=%s)",
            len(results),
            f"{top_score:.3f}" if top_score is not None else "none",
            query[:120],
            hindi,
        )

        if not results or (top_score is not None and top_score < self._abstain_threshold):
            return self._abstain(query, results, started, hindi)

        # Phase 2: ensure current law (BNS/BNSS/BSA) is in context for any repealed
        # reference in the question or the retrieved sources, so we can lead with it.
        results = self._expand_current_law(results, search_query)
        ordered = _order_for_context(results)

        visible = ordered[:_CONTEXT_CHUNKS]
        context = _format_context(visible)
        # For a criminal matter, inject the BNSS process arc as a SEPARATE context block so
        # "what happens next" can be grounded without crowding the offence sections out of
        # the main window. Empty (and harmless) for non-criminal matters.
        proc_chunks = self._procedure_context(visible)
        # For a follow-up, answer the resolved standalone question (so "that" is concrete);
        # for a first turn, keep the user's original phrasing.
        prompt_question = search_query if history else query
        prompt = PROMPT_TEMPLATE.format(context=context, query=prompt_question)
        if proc_chunks:
            prompt += (
                '\n\n---\nPROCEDURE CONTEXT (BNSS — how a criminal case proceeds; use ONLY '
                'to fill the "what_happens_next" steps, each step naming its BNSS section):\n'
                + _format_context(proc_chunks)
            )
        if hindi:
            prompt += _HINDI_INSTRUCTION
        raw = self._llm.generate_json(prompt)

        answer = _stringify(raw.get("answer")) or (
            _FALLBACK_ANSWER_HI if hindi else
            "Typically under Indian law, this question needs a closer look at the facts and provisions."
        )
        law_reference = _stringify(raw.get("law_reference")) or "General Indian law"
        action = _stringify(raw.get("action")) or (
            _FALLBACK_ACTION_HI if hindi else
            "Consult a qualified lawyer for guidance specific to your situation."
        )
        reasoning = _stringify(raw.get("reasoning")) or "No strong context, used general principles."

        # Reporter-citation scrub: never surface a precise law-report citation string
        # (SCC/SCR/AIR) that the LLM appended but that isn't present verbatim in a source
        # the user can actually see. Graded against EXACTLY the visible source chunks (the
        # citations the answer is allowed to lean on) + the injected procedure arc.
        scrub_sources = visible + proc_chunks
        answer = _scrub_reporter_citations(answer, scrub_sources)
        reasoning = _scrub_reporter_citations(reasoning, scrub_sources)

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

        # Grave-offence current-law guard: a headline that is internally CONTRADICTED by the
        # curated repealed->current map (e.g. "BNS 307 (was IPC 307)" — IPC 307's successor is
        # BNS 109, a theft section being asserted as attempt-to-murder) is a misclassification
        # signal as serious as an unverified citation. We do NOT silently rewrite the offence
        # (that risks asserting the wrong section the other way); we downgrade to low confidence
        # so the rich analysis block (which needs high confidence) is SUPPRESSED and the citizen
        # is escalated to legal aid — failing safe rather than vouching for a contradictory map.
        guard = _current_law_guard_violation(law_reference)
        if guard:
            logger.warning(
                "Current-law guard tripped: headline %r contradicts %s->%s mapping — downgrading.",
                law_reference, guard[2], guard[0],
            )
            confidence = "low"

        current_law_note = self._current_law_note(ordered, search_query, hindi, law_reference)
        citations = self._dedupe_citations(ordered)
        escalation = None
        if confidence == "low" or not citation_verified:
            escalation = LEGAL_AID_ESCALATION_HI if hindi else LEGAL_AID_ESCALATION

        # Rich case-analysis — built only on a strong, verified answer, and graded against
        # EXACTLY the chunks the model saw (offence context + injected procedure arc).
        analysis = self._build_analysis(
            raw.get("analysis"), visible, proc_chunks, hindi, confidence, citation_verified, law_reference
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
            analysis=analysis,
            disclaimer=DISCLAIMER_HI if hindi else DISCLAIMER,
            response_time_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _merge_results(*result_lists: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Union retrieval result sets, de-duped by chunk id, keeping the MAX score seen
        for each chunk; sorted best-first. Used to combine the rewrite and original-query
        retrievals so the strongest evidence for a chunk wins regardless of which phrasing
        surfaced it."""
        best: dict[str, RetrievedChunk] = {}
        for results in result_lists:
            for rc in results:
                cid = rc.chunk.id
                kept = best.get(cid)
                if kept is None or rc.score > kept.score:
                    best[cid] = rc
        return sorted(best.values(), key=lambda rc: rc.score, reverse=True)

    def _robust_retrieve(
        self, query: str, search_query: str, history: list, hindi: bool
    ) -> tuple[list[RetrievedChunk], str]:
        """Retrieve robustly against query-rewrite variance.

        The first attempt uses the rewrite already computed by the caller. If its top
        rerank score clears the abstain bar, we're done (the fast, common path — no extra
        work). Otherwise the rewrite *may* simply be a weak sampling of a non-deterministic
        rewriter (a lay consumer narrative reranks deeply negative raw, and only a good
        rewrite rescues it). So we make ONE cheap recovery pass with at most one extra
        rewrite:
          - retrieve with the ORIGINAL text (free of rewrite variance, no LLM call), and
          - draw ONE STRONG recovery rewrite (_recovery_query) and retrieve with it,
        then keep the best-scoring MERGED set (union by chunk id, max score). Because the
        merge keeps the max score per chunk across attempts, if ANY phrasing clears the bar
        the merged top clears it — removing the ~50/50 flakiness WITHOUT lowering the
        (correctly calibrated) abstain threshold. A genuinely off-topic query has no
        phrasing that reranks above 0 (and _recovery_query returns the original text for a
        non-legal query, so we never manufacture a statute match for junk) — junk keeps
        abstaining on every run.

        Returns the chosen (results, search_query) so the caller can keep prompting with the
        phrasing that actually retrieved.
        """
        results = self._retriever.retrieve(search_query)
        top = results[0].score if results else None
        if top is not None and top >= self._abstain_threshold:
            return results, search_query

        logger.info(
            "Rewrite retrieval weak (top=%s < %.2f) - recovering with original + strong recovery rewrite",
            f"{top:.3f}" if top is not None else "none",
            self._abstain_threshold,
        )

        attempts: list[tuple[list[RetrievedChunk], str]] = [(results, search_query)]

        # 1. The original text, with no rewrite variance at all. Cheap and deterministic.
        try:
            orig = self._retriever.retrieve(query)
            attempts.append((orig, query))
        except Exception as e:
            logger.warning("Original-query retrieval failed during recovery: %s", e)

        # 2. One STRONG recovery rewrite. Rather than just re-sampling the same prompt (which
        #    flakes the same way), use a stricter prompt engineered to LEAD with the governing
        #    Act and STACK its formal terms — the phrasing shape that reranks well and stably
        #    (a "Consumer Protection Act deficiency in service defective goods unfair trade
        #    practice consumer complaint"-style query scores ~+3, not the lay narrative's -6).
        try:
            alt = self._recovery_query(query, history, hindi)
            if alt.strip() not in {search_query.strip(), query.strip()}:
                alt_results = self._retriever.retrieve(alt)
                attempts.append((alt_results, alt))
        except Exception as e:
            logger.warning("Recovery-rewrite retrieval failed: %s", e)

        # Keep the attempt whose own top score is highest, but MERGE every attempt into it so
        # a chunk that scored well under any phrasing is available (max score per chunk id).
        def attempt_top(a: tuple[list[RetrievedChunk], str]) -> float:
            return a[0][0].score if a[0] else float("-inf")

        best_attempt = max(attempts, key=attempt_top)
        merged = self._merge_results(*[a[0] for a in attempts])
        merged = merged[: self._retriever.top_k]
        new_top = merged[0].score if merged else None
        logger.info(
            "Recovery: best single top=%s, merged top=%s (chose query=%r)",
            f"{attempt_top(best_attempt):.3f}" if best_attempt[0] else "none",
            f"{new_top:.3f}" if new_top is not None else "none",
            best_attempt[1][:80],
        )
        return merged, best_attempt[1]

    # ------------------------------------------------------------------ #
    def _procedure_context(self, visible: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """For a criminal matter, fetch the canonical BNSS process-arc sections so the
        case analysis can ground 'what happens next' in real procedure text. Offence
        queries never retrieve procedure on their own. Best-effort: [] for a non-criminal
        matter or on any failure (the section then simply stays hidden)."""
        try:
            # Criminal signal: a BNS (penal code) offence is in the visible context.
            if not any((rc.chunk.act or "").upper() == "BNS" for rc in visible):
                return []
            already = {
                _base_num(rc.chunk.section or "")
                for rc in visible
                if (rc.chunk.act or "").upper() == "BNSS"
            }
            wanted = [s for s in _BNSS_PROCESS_ARC if s not in already]
            if not wanted:
                return []
            from app.rag.vector_store import VectorStore

            chunks = VectorStore.instance().fetch_by_reference(act="BNSS", sections=wanted, limit=len(wanted))
            order = {s: i for i, s in enumerate(_BNSS_PROCESS_ARC)}
            out = [RetrievedChunk(chunk=c, score=0.0) for c in chunks]
            out.sort(key=lambda rc: order.get(_base_num(rc.chunk.section or ""), 99))
            if out:
                logger.info("Injected %d BNSS process-arc section(s) for 'what happens next'", len(out))
            return out
        except Exception as e:  # never let procedure injection break a query
            logger.warning("Procedure-context injection skipped: %s", e)
            return []

    # ------------------------------------------------------------------ #
    @staticmethod
    def _offence_classification(law_reference: str, hindi: bool) -> str:
        """A grounded one-line offence classification (BNSS First Schedule) for the PRIMARY
        (lead) cited BNS offence ONLY.

        Returns "" when the lead offence has no unambiguous First-Schedule entry, or is a
        death-deeming offence (_NO_AUTO_CLASS). Classifying ONLY the lead — never walking on
        to a secondary section — is deliberate: the previous "first groundable section"
        behaviour surfaced a lesser/secondary offence's lighter class whenever the real lead
        offence was ambiguous (e.g. voyeurism 77 -> intimidation 351's "non-cognizable,
        bailable") or fell through an ambiguous cruelty 85 to dowry-death 80. A conditional
        or ambiguous lead now stays UNLABELLED rather than mislabelled."""
        try:
            # Grave-offence current-law guard: if the headline is internally inconsistent
            # with the curated repealed->current map (the IPC 307 -> BNS 307 attempt-on-life
            # footgun), suppress the grounded classification rather than assert a label for a
            # section the answer evidently cited by mistake.
            if _current_law_guard_violation(law_reference):
                logger.info("Offence classification suppressed: current-law guard tripped for %r", law_reference)
                return ""
            secs = re.findall(
                r"\bBNS\b\s*(?:Section|Sec\.?|S\.?|धारा)?\s*(\d+[A-Za-z]?)",
                law_reference or "", re.IGNORECASE,
            )
            if not secs:
                return ""
            lead = secs[0]
            if _base_num(lead) in _NO_AUTO_CLASS:
                return ""
            from app.rag.offence_classification import OffenceClassification

            return OffenceClassification.instance().describe(lead, hindi, name=True) or ""
        except Exception as e:
            logger.warning("Offence classification skipped: %s", e)
            return ""

    # ------------------------------------------------------------------ #
    def _build_analysis(
        self,
        raw_analysis: object,
        visible: list[RetrievedChunk],
        proc: list[RetrievedChunk],
        hindi: bool,
        confidence: Confidence,
        citation_verified: bool,
        law_reference: str,
    ) -> CaseAnalysis | None:
        """Turn the LLM's "analysis" object into a trust-safe CaseAnalysis — or None.

        Master gate first: a rich block is earned ONLY by a high-confidence (Mode A),
        citation-verified answer whose OWN cited section is present in the context the model
        actually saw. Anything weaker → None (the citizen sees today's honest paragraph).
        Then every bullet is deterministically scrubbed (drops ungrounded offence labels,
        second-person outcome predictions, and any precedent-shaped token), and section-
        citing bullets are re-checked against the visible context. Never raises."""
        try:
            if not isinstance(raw_analysis, dict):
                return None
            # --- MASTER SUPPRESSION GATE ---
            if confidence != "high" or not citation_verified:
                return None
            ref = (law_reference or "").lower()
            is_generic = (not ref) or any(g in ref for g in _GENERIC_REFS) or "article" in ref
            if is_generic:
                return None  # a generic/Mode-B headline never earns analysis
            available = _available_sections(visible)
            if not available:
                return None
            cited = {_base_num(t).upper() for t in _NUM_RE.findall(law_reference)}
            cited.discard("")
            if not (cited & available):
                return None  # the headline's own section must be in the visible context

            # Procedure steps ground against the injected BNSS process arc too (offence
            # queries don't retrieve procedure, so without this 'what happens next' is empty).
            bnss = _bnss_sections(visible) | _bnss_sections(proc)

            def clean(key: str) -> list[str]:
                """Trim, cap, and drop ungrounded-classification / outcome-prediction bullets."""
                items = raw_analysis.get(key)
                if not isinstance(items, list):
                    return []
                out: list[str] = []
                for it in items:
                    s = _stringify(it)
                    if not s:
                        continue
                    s = s[:_ANALYSIS_MAX_LEN].strip()
                    if _CLASSIFICATION_RE.search(s) or _OUTCOME_PREDICTION_RE.search(s):
                        continue
                    out.append(s)
                    if len(out) >= _ANALYSIS_MAX_ITEMS:
                        break
                return out

            def cited_nums(s: str) -> set[str]:
                nums = {_base_num(t).upper() for t in _NUM_RE.findall(s)}
                nums.discard("")
                return nums

            situation = clean("situation")
            do_now = clean("do_now")

            # applicable_law: a bullet that names sections must have one present in context.
            applicable_law = [
                s for s in clean("applicable_law")
                if not cited_nums(s) or (cited_nums(s) & available)
            ]
            # what_happens_next: each step must name a BNSS section that's in the context;
            # otherwise drop it (no generic procedure dressed up as this user's case).
            what_happens_next = [s for s in clean("what_happens_next") if cited_nums(s) & bnss]
            # also_possible: allowlist — keep only impersonal "the law / a court may…" bullets.
            also_possible = [
                s for s in clean("also_possible")
                if any(stem in s.lower()[:32] for stem in _SAFE_STEMS)
            ]
            # for_your_advocate: never let a precedent-shaped citation through.
            for_your_advocate = [s for s in clean("for_your_advocate") if not _PRECEDENT_RE.search(s)]

            if not any([situation, applicable_law, what_happens_next, do_now, also_possible, for_your_advocate]):
                return None  # nothing survived scrubbing → fall back to the plain card

            classification = self._offence_classification(law_reference, hindi)

            return CaseAnalysis(
                outcome_framing=_OUTCOME_FRAMING_HI if hindi else _OUTCOME_FRAMING,
                classification=classification,
                situation=situation,
                applicable_law=applicable_law,
                what_happens_next=what_happens_next,
                do_now=do_now,
                also_possible=also_possible,
                for_your_advocate=for_your_advocate,
            )
        except Exception as e:  # never let analysis break a query — degrade to the plain card
            logger.warning("Case-analysis build skipped: %s", e)
            return None

    # ------------------------------------------------------------------ #
    def _standalone_query(self, query: str, history: list, hindi: bool) -> str:
        """Resolve the user's question into a standalone English search query.

        Three jobs, so retrieval against the English corpus stays strong:
          - condense a follow-up using recent history ('what's the punishment for that?'),
          - translate a Hindi question, and
          - distil a lay NARRATIVE ('he took my money to invest and won't return it') into
            the legal concepts that actually retrieve ('cheating, criminal breach of trust').
        A described situation retrieves poorly verbatim — that's exactly when the case
        analysis should fire, so we must rewrite it. A short English keyword question is
        already a good query and skips the LLM call (fast single-turn path). Falls back to
        the original text on any failure."""
        has_devanagari = bool(_DEVANAGARI.search(query))
        # A described situation is long and lay-worded; a keyword lookup is short. Only the
        # former needs distilling, so the common short-question path stays a single call.
        # A lay narrative isn't always >=12 words ("the shopkeeper sold me a defective mixer
        # and refuses to refund" is 11) — yet it retrieves DEEPLY negative verbatim because
        # it names no statute. So we also distil a SHORTER text that *reads* like a narrative:
        # first/third-person storytelling (I/me/my/we, or "the <party> <verb>ed me/my…"). A
        # terse keyword lookup ("knife attack ICU", "grievous hurt punishment") has no such
        # signal and still skips the LLM (fast path). Junk that happens to match ("which
        # smartphone should I buy") is harmless: it reranks far below the abstain bar either
        # way, so distilling it never makes it answer.
        words = query.split()
        is_narrative = len(words) >= 12 or (len(words) >= 7 and _is_lay_narrative(query))
        if not history and not has_devanagari and not is_narrative:
            return query
        try:
            if history:
                convo = "\n".join(
                    f"{t.role}: {t.content[:500]}" for t in history[-6:] if getattr(t, "content", "")
                )
                prompt = (
                    "Rewrite the user's LATEST message as a single standalone English legal "
                    "search query naming the likely offence(s) or legal concept(s). Use the "
                    "conversation to resolve references like 'that', 'it', or 'the punishment'. "
                    'Return ONLY JSON: {"q": "..."}.\n\nCONVERSATION:\n' + convo + "\nLATEST: " + query
                )
            else:
                prompt = (
                    "Rewrite this into a short English legal search query (4-15 words) that a "
                    "statute book or case index would match. Name the most relevant Indian "
                    "Act/code AND the precise legal concept(s) — across ALL areas of law, not "
                    "just crimes:\n"
                    "- consumer (defective goods, deficient service, refund, non-delivery, "
                    "warranty, e-commerce) => e.g. 'Consumer Protection Act deficiency in "
                    "service defective goods unfair trade practice consumer complaint';\n"
                    "- crime => the offence + code, e.g. 'cheating BNS', 'grievous hurt';\n"
                    "- tenancy/labour/family/civil => the cause of action and its statute.\n"
                    "Prefer formal statutory terms over lay words. Translate from Hindi if "
                    'needed. Return ONLY JSON: {"q": "..."}.'
                    "\n\nTEXT: " + query
                )
            q = _stringify(self._llm.generate_json(prompt).get("q"))
            if q:
                logger.info("Resolved query for retrieval: %r -> %r", query[:60], q[:80])
                return q
        except Exception as e:
            logger.warning("Query resolution failed (%s); retrieving with original text.", e)
        return query

    # ------------------------------------------------------------------ #
    def _recovery_query(self, query: str, history: list, hindi: bool) -> str:
        """A STRONG, statute-leading rewrite used only when the first rewrite retrieved
        below the abstain bar.

        The default rewrite asks for a terse 4-15-word query, which on a lay narrative
        sometimes omits the governing Act and reranks negative. This recovery prompt instead
        forces the shape that reranks high and stably: LEAD with the exact Indian Act/code,
        then STACK its formal statutory terms (the corpus is statute text, so term overlap is
        what the cross-encoder rewards). Falls back to the default rewrite, then the original
        text, on any failure — never raises."""
        try:
            convo = ""
            if history:
                convo = "\n\nCONVERSATION (resolve 'that'/'it'/'the punishment' from it):\n" + "\n".join(
                    f"{t.role}: {t.content[:300]}" for t in history[-6:] if getattr(t, "content", "")
                )
            prompt = (
                "You convert a person's LEGAL PROBLEM into one strong English search query "
                "for an Indian STATUTE database.\n"
                "FIRST decide: does the text describe an actual legal matter — a dispute, a "
                "harm, a right, an offence, a contract/grievance someone could take to a court, "
                "tribunal, police, or consumer forum?\n"
                "- If NO (it is shopping advice, a product recommendation, a recipe, general "
                "chit-chat, or anything with no legal grievance), return {\"q\": \"\"} — an "
                "EMPTY string. Do NOT invent a law for a non-legal question.\n"
                "- If YES, build the query: (1) START with the exact governing Indian Act/code "
                "name in full (e.g. 'Consumer Protection Act', 'Bharatiya Nyaya Sanhita', "
                "'Transfer of Property Act', 'Negotiable Instruments Act'); (2) THEN list 5-9 "
                "formal statutory terms — NOT lay words. Consumer example: 'Consumer Protection "
                "Act deficiency in service defective goods unfair trade practice consumer "
                "complaint compensation refund'. Crime example: 'Bharatiya Nyaya Sanhita "
                "cheating dishonest inducement delivery of property'. No sentences, no "
                "pronouns, no story — just the Act name followed by terms.\n"
                "Translate from Hindi if needed. Return ONLY JSON: {\"q\": \"...\"}."
                + convo + "\n\nTEXT: " + query
            )
            q = _stringify(self._llm.generate_json(prompt).get("q"))
            if q:
                logger.info("Recovery rewrite: %r -> %r", query[:60], q[:90])
                return q
            # Empty q == the model judged this not a legal matter. Honour that: do NOT fall
            # through to a rewrite that would manufacture a statute query for junk. Return the
            # original so junk keeps reranking far below the abstain bar (stays abstained).
            logger.info("Recovery rewrite judged non-legal (empty q) for %r — keeping original.", query[:60])
            return query
        except Exception as e:
            logger.warning("Recovery rewrite failed (%s); falling back to default rewrite.", e)
        # Fall back to a default rewrite (still better than the original lay narrative).
        return self._standalone_query(query, history, hindi)

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

    def _current_law_note(
        self, results: list[RetrievedChunk], query: str, hindi: bool = False, law_reference: str = ""
    ) -> str | None:
        """Bridge old<->current law, anchored to the section actually cited.

        Anchoring to the headline citation (not just the first repealed ref in retrieval)
        keeps the note relevant: an answer citing BNS 103 gets the IPC 302->BNS 103 bridge,
        not whatever unrelated repealed section happened to be retrieved alongside it."""
        from app.rag.law_map import LawMap

        law_map = LawMap.instance()
        from_codes = {c.upper() for c in law_map.from_codes()}
        to_codes = {c.upper() for c in law_map.to_codes()}

        def build(code: str, sec: str) -> str | None:
            note = law_map.current_reference_note(code, sec, hindi=hindi)
            if note and not law_map.verified_for(code):
                note += (
                    " (यह मानचित्रण सांकेतिक है — आधिकारिक मूल अधिनियम से पुष्टि करें।)" if hindi
                    else " (Mapping is indicative — confirm against the official bare act.)"
                )
            return note

        # 1. Anchor to the headline citation.
        m = re.match(r"\s*([A-Za-z]+)\D*(\d+[A-Za-z]?)", law_reference or "")
        if m:
            code, sec = m.group(1).upper(), m.group(2)
            if code in from_codes:                       # cited a repealed section -> bridge forward
                note = build(code, sec)
                if note:
                    return note
            elif code in to_codes:                       # cited a current section -> bridge from its predecessor
                pred = law_map.predecessor_ref(code, sec)
                if pred and (note := build(pred[0], pred[1])):
                    return note

        # 2. Fallback: the first repealed reference anywhere in the question/sources.
        for code, sec in self._collect_repealed_refs(results, query, list(law_map.from_codes())):
            note = build(code, sec)
            if note:
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
    def _abstain(
        self, query: str, results: list[RetrievedChunk], started: float, hindi: bool = False
    ) -> AskResponse:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info("Abstained (weak retrieval) in %dms for query=%r (hindi=%s)", elapsed_ms, query[:120], hindi)
        return AskResponse(
            answer=_ABSTAIN_ANSWER_HI if hindi else (
                "I couldn't find a reliable basis in my legal sources to answer this "
                "confidently. To avoid giving you wrong legal information, I'd rather not "
                "guess. Please rephrase with more detail, or seek the help below."
            ),
            law_reference="General Legal Guidance",
            action=_ABSTAIN_ACTION_HI if hindi else
            "Contact free legal aid or a qualified lawyer for your specific situation.",
            confidence="low",
            reasoning="No strong context, used general principles.",
            # Abstaining means nothing scored as reliably relevant — don't dangle the
            # weak/below-threshold chunks as if they were sources for an answer.
            citations=[],
            abstained=True,
            escalation=LEGAL_AID_ESCALATION_HI if hindi else LEGAL_AID_ESCALATION,
            current_law_note=None,
            citation_verified=True,
            disclaimer=DISCLAIMER_HI if hindi else DISCLAIMER,
            response_time_ms=elapsed_ms,
        )
