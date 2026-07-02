# -*- coding: utf-8 -*-
"""Build data/distill/seed_queries.jsonl — the seed-query set for distillation.

Fully deterministic (NO LLM calls). Mix (Gate-1 corrective — OWN_MODEL_PLAN.md §4.1):

    indicqa      20%   consumer-grade questions derived from the IndicLegalQA dataset
    narrative_en 35%   hand-authored lay-narrative templates (10 audit domains) + slots
    narrative_hi 25%   hand-authored Devanagari templates (UTF-8 literals in THIS file)
    followup     15%   two-turn seeds (narrative turn-1 + static assistant reply + follow-up)
    thin          5%   legal-ish weak-context queries (obscure / state-specific / vague)

Shared seed contract (one JSON per line):
    {"id": "<sha1-12 of query|language|json(history)>", "query": str,
     "language": "en"|"hi", "history": [{"role","content"}...] (empty if single-turn),
     "domain": str, "kind": "indicqa"|"narrative_en"|"narrative_hi"|"followup"|"thin"}

The id convention matches the sibling generator's smoke seeds exactly:
    sha1(f"{query}|{language}|{json.dumps(history, ensure_ascii=False)}")[:12]

indicqa design note (based on actually reading the dataset):
    'IndicLegalQA Dataset_10K.json' is a list of 10,002 {case_name, judgment_date,
    question, answer} rows. ~95% of the questions are case-recall trivia ("What was the
    Supreme Court's final decision in the case?", "Who is the respondent in X vs. Y?").
    A strict multi-signal filter (case anchors, party roles, deictic pronouns,
    proper-name runs, past-tense recall starters) leaves only ~100-200 questions that a
    consumer could ask standalone. To honour the 40% quota WITHOUT admitting trivia, the
    indicqa pool therefore has two deterministic sub-sources, both derived purely from
    the dataset:
      (a) verbatim: dataset questions that survive the strict consumer-grade filter
          (after stripping "...in the context of this case" style suffixes), and
      (b) mined: canonical consumer question forms generated from the (section, act) /
          (article) / act-level legal references that the dataset's own questions cite
          (e.g. "Section 37 of the NDPS Act" -> "What does Section 37 of the Narcotic
          Drugs and Psychotropic Substances Act, 1985 deal with?"), most-cited first.
    If a pool still cannot fill its quota after golden-set exclusion + dedupe, the
    shortfall is redistributed to the other kinds (with a printed WARNING).

Exclusions: every candidate whose normalized query (or follow-up turn-1 narrative)
exactly matches, contains, or is contained in a normalized data/eval/golden.jsonl query
is dropped — the eval set must never leak into training seeds.

Hindi gotcha: all Devanagari lives inside this file / the output JSONL as UTF-8.
Never pipe it through shell echo/vars.

Usage (from the backend root):
    .venv/Scripts/python.exe scripts/distill/build_seed_queries.py --total 2400
    .venv/Scripts/python.exe scripts/distill/build_seed_queries.py --total 2400 --seed 7 --show-samples 3
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

try:  # Windows consoles default to a legacy codepage; Devanagari needs UTF-8.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

DEFAULT_OUT = ROOT / "data" / "distill" / "seed_queries.jsonl"
INDICQA_JSON = ROOT / "data" / "indiclegalqa" / "IndicLegalQA Dataset_10K.json"
GOLDEN_JSONL = ROOT / "data" / "eval" / "golden.jsonl"

# Gate-1 corrective mix (see OWN_MODEL_PLAN.md §4.1). The 100-seed pilot MEASURED
# clean-keep rates: followup 10/10, narrative_en 84%, narrative_hi 60%, indicqa 47.5%,
# thin 20% (by design). indicqa is down-weighted both for its low clean yield AND the
# three-way collision (it is already 82% of the retrieval corpus and eval-adjacent);
# the high-yield behaviors closest to real citizen usage are up-weighted. Projected
# clean ≈70%; ~2,900 seeds → ~2,000 clean pairs ≈ $19.
MIX = [  # (kind, fraction) — order also fixes largest-remainder tie-breaks
    ("indicqa", 0.20),
    ("narrative_en", 0.35),
    ("narrative_hi", 0.25),
    ("followup", 0.15),
    ("thin", 0.05),
]

# The 10 audit domains (+ "other" fallback for indicqa/thin strays).
DOMAINS = [
    "violent_crime", "property_fraud", "family_dowry_dv", "police_procedure",
    "consumer", "labour", "tenancy", "documents_rti", "evidence", "accident_mva",
]


# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #
def sha1_12(query: str, language: str, history: list[dict]) -> str:
    payload = f"{query}|{language}|{json.dumps(history, ensure_ascii=False)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def norm(s: str) -> str:
    """Normalization for dedupe/golden-exclusion: lowercase, strip punctuation
    (keeping Devanagari), collapse whitespace."""
    s = s.lower()
    s = re.sub(r"[^0-9a-zऀ-ॿ]+", " ", s)
    return " ".join(s.split())


def make_seed(query: str, language: str, history: list[dict], domain: str, kind: str) -> dict:
    return {
        "id": sha1_12(query, language, history),
        "query": query,
        "language": language,
        "history": history,
        "domain": domain,
        "kind": kind,
    }


def quotas(total: int) -> dict[str, int]:
    """Largest-remainder apportionment so the per-kind quotas sum to exactly total."""
    base = {k: int(math.floor(total * f)) for k, f in MIX}
    remainders = sorted(
        ((total * f - math.floor(total * f), i, k) for i, (k, f) in enumerate(MIX)),
        key=lambda t: (-t[0], t[1]),
    )
    for _, _, k in remainders[: total - sum(base.values())]:
        base[k] += 1
    return base


def expand(template: str, slots: dict[str, list[str]], rng: random.Random) -> list[str]:
    """All slot combinations of a template, deterministically shuffled."""
    if not slots:
        return [template]
    names = sorted(slots)
    out = [template.format(**dict(zip(names, combo)))
           for combo in itertools.product(*(slots[n] for n in names))]
    rng.shuffle(out)
    return out


def round_robin(pools: list[list]) -> list:
    """Interleave pools (one item from each in turn) until all are exhausted —
    keeps the selected quota spread across templates instead of exhausting one."""
    out, idx = [], [0] * len(pools)
    remaining = sum(len(p) for p in pools)
    while remaining:
        for i, p in enumerate(pools):
            if idx[i] < len(p):
                out.append(p[idx[i]])
                idx[i] += 1
                remaining -= 1
    return out


class SeedSelector:
    """Applies golden-set exclusion + global dedupe while filling quotas."""

    def __init__(self, golden_norms: list[str]) -> None:
        self.golden = golden_norms
        self.seen_keys: set[str] = set()
        self.seen_ids: set[str] = set()
        self.excluded_golden = 0
        self.excluded_dupe = 0

    def _golden_like(self, text: str) -> bool:
        nq = norm(text)
        if not nq:
            return False
        return any(g == nq or g in nq or nq in g for g in self.golden)

    def accept(self, seed: dict) -> bool:
        texts = [seed["query"]] + [t["content"] for t in seed["history"] if t.get("role") == "user"]
        if any(self._golden_like(t) for t in texts):
            self.excluded_golden += 1
            return False
        key = norm(seed["query"]) + "||" + norm(" ".join(t["content"] for t in seed["history"]))
        if key in self.seen_keys or seed["id"] in self.seen_ids:
            self.excluded_dupe += 1
            return False
        self.seen_keys.add(key)
        self.seen_ids.add(seed["id"])
        return True

    def take(self, candidates: list[dict], quota: int) -> tuple[list[dict], list[dict]]:
        """Return (chosen up to quota, untouched leftovers for redistribution)."""
        chosen: list[dict] = []
        i = 0
        for i, seed in enumerate(candidates, start=1):
            if len(chosen) >= quota:
                i -= 1
                break
            if self.accept(seed):
                chosen.append(seed)
        return chosen, candidates[i:]


# --------------------------------------------------------------------------- #
# Domain tagging from keywords
# --------------------------------------------------------------------------- #
_DOMAIN_KEYWORDS: list[tuple[str, re.Pattern]] = [
    ("accident_mva", re.compile(r"\b(motor vehicle|road accident|hit[- ]and[- ]run|rash|negligent driv|driving|mact|third[- ]party insurance)\b", re.I)),
    ("family_dowry_dv", re.compile(r"\b(dowry|498-?a|304-?b|domestic violence|divorce|maintenance|custody of|alimony|marriage|husband|wife|in-?laws|cruelty)\b", re.I)),
    ("police_procedure", re.compile(r"\b(fir|bail|anticipatory|arrest|remand|police custody|charge ?sheet|summons|warrant|investigation|magistrate|cr\.?p\.?c|criminal procedure|bnss)\b", re.I)),
    ("violent_crime", re.compile(r"\b(murder|homicide|hurt|assault|kidnap|abduct|acid|rape|molest|stalking|intimidation|attempt to murder|grievous|outrag)\b", re.I)),
    ("property_fraud", re.compile(r"\b(theft|stolen|robbery|dacoity|cheat|fraud|forgery|extortion|breach of trust|cheque|negotiable instruments|cyber|online scam|otp|hacking|misappropriat)\b", re.I)),
    ("consumer", re.compile(r"\b(consumer|deficiency (of|in) service|refund|defective|warranty|unfair trade)\b", re.I)),
    ("labour", re.compile(r"\b(wages|salary|gratuity|provident fund|\besi\b|retrenchment|workman|industrial dispute|bonus|overtime|employer|employee|labour)\b", re.I)),
    ("tenancy", re.compile(r"\b(rent|landlord|tenant|eviction|lease|security deposit)\b", re.I)),
    ("documents_rti", re.compile(r"\b(rti|right to information|certificate|affidavit|\bwill\b|succession|registration|stamp duty|notary|mutation|sale deed|gift deed)\b", re.I)),
    ("evidence", re.compile(r"\b(evidence|witness|dying declaration|electronic record|testimony|admissib|65-?b|cctv)\b", re.I)),
]


def keyword_domain(text: str) -> str:
    for dom, pat in _DOMAIN_KEYWORDS:
        if pat.search(text):
            return dom
    return "other"


# --------------------------------------------------------------------------- #
# KIND 1 — indicqa (verbatim survivors + reference-mined canonical questions)
# --------------------------------------------------------------------------- #
# strip "…, and how does it apply to this case?" style suffixes / infixes
_STRIP_SUFFIX = re.compile(
    r"(,?\s*(and\s+)?(how|why)\s+(does|do|did|is|was)\s+(it|this|that|these|they)\b[^?]*"
    r"|,?\s*in\s+(the\s+context\s+of\s+)?(this|that|the\s+present)\s+(case|matter|dispute|situation|judgment)[^?]*"
    r"|,?\s*as\s+(discussed|mentioned|noted|highlighted|applied|interpreted)[^?]*"
    r"|,?\s*according\s+to\s+the\s+(judgment|court|supreme\s+court)[^?]*)\?\s*$",
    re.I,
)
_STRIP_PREFIX = re.compile(r"^(predictive|clarifying|hypothetical|reflective|analytical)\s*:\s*", re.I)

_CASE_ANCHOR = re.compile(
    r"(which case|th(e|is|at) case\b|present case|\bvs\.?|\bversus\b|\bv\.\s|case (no|crime)"
    r"|crime no|fir no|appeal no|petition no|civil appeal|criminal appeal|\bslp\b|writ petition"
    r"|\bthe (judgment|verdict|ruling|order|decision|award|decree|appeal|petition|suit|trial"
    r"|proceedings|incident|dispute|matter|fir|chargesheet|charge sheet|impugned)\b"
    r"|\bthis (judgment|decision|ruling|appeal|petition|order)\b)",
    re.I,
)
_PARTY_WORDS = re.compile(
    r"\b(appellants?|respondents?|petitioners?|accused|complainants?|deceased|prosecution"
    r"|prosecutrix|plaintiffs?|defendants?|detenu|assessees?|informant|witnesses?"
    r"|pw-?\d+|his|her|he|she|him|their|they|them)\b",
    re.I,
)
_PAST_RECALL = re.compile(r"\b(was|were|did|had)\b", re.I)
_PRACTITIONER = re.compile(r"(legal practitioners|similar allegations|similar cases)", re.I)
_PREDICTIVE = re.compile(r"^(what|how)\s+(might|could)\b", re.I)
_DEICTIC_THIS = re.compile(r"\bthis\b(?!\s+(section|act|article|provision|rule|law|code|offence))", re.I)
_LAW_SUBSTANCE = re.compile(
    r"(section\s+\d|article\s+\d|\b(act|code|ipc|crpc|cpc|penal|constitution)\b"
    r"|\b(bail|arrest|remand|custody|maintenance|divorce|dowry|cheque|limitation"
    r"|adverse possession|compensation|dying declaration|cruelty|retrenchment|gratuity"
    r"|wages|tenanc|eviction|consumer|negligence|defamation|anticipatory|parole"
    r"|furlough|probation|legal aid|evidence)\b)",
    re.I,
)
# capitalized runs of 2+ significant words with NO statutory/institutional word -> party name
_NAME_RUN = re.compile(r"\b([A-Z][A-Za-z.'\-]*(?:\s+(?:of|and|&|the|for|[A-Z][A-Za-z.'\-]*))+)\b")
_NAME_WHITELIST = {
    "act", "acts", "code", "codes", "rule", "rules", "regulation", "regulations", "order",
    "constitution", "amendment", "schedule", "court", "courts", "tribunal", "commission",
    "board", "authority", "ministry", "government", "union", "state", "states", "india",
    "indian", "penal", "procedure", "criminal", "civil", "evidence", "contract", "companies",
    "income", "tax", "service", "services", "police", "scheme", "fund", "provident",
    "insurance", "motor", "vehicles", "negotiable", "instruments", "consumer", "protection",
    "domestic", "violence", "hindu", "muslim", "marriage", "succession", "transfer",
    "property", "registration", "stamp", "limitation", "arbitration", "conciliation",
    "specific", "relief", "industrial", "disputes", "wages", "gratuity", "bonus",
    "factories", "information", "technology", "pocso", "ndps", "ipc", "crpc", "cpc",
    "tada", "pota", "uapa", "pmla", "fema", "rbi", "sebi", "esi", "fssa", "gst", "vat",
    "central", "section", "sections", "article", "articles", "supreme", "high", "national",
    "legal", "aid", "education", "right", "rights", "land", "acquisition", "food", "safety",
    "security", "wildlife", "forest", "environment", "pollution", "water", "air", "mines",
    "minerals", "electricity", "railways", "customs", "excise", "goods", "sales", "entry",
    "building", "urban", "rural", "municipal", "panchayat", "cooperative", "societies",
    "trusts", "wakf", "waqf", "endowments", "religious", "charitable", "juvenile",
    "justice", "care", "children", "women", "dowry", "prohibition", "prevention",
    "corruption", "money", "laundering", "benami", "transactions", "essential",
    "commodities", "arms", "explosives", "passports", "citizenship", "foreigners",
    "representation", "people", "contempt", "advocates", "general", "clauses", "public",
    "premises", "standards", "employees", "employment", "compensation", "banking",
    "insolvency", "bankruptcy", "securities", "contracts", "partnership", "trade",
    "marks", "copyright", "patents", "designs",
}


def _has_party_name(q: str) -> bool:
    for m in _NAME_RUN.finditer(q):
        words = [w.lower().strip(".,'") for w in m.group(1).split()]
        significant = [w for w in words if w not in ("of", "and", "&", "the", "for")]
        if len(significant) >= 2 and not any(w in _NAME_WHITELIST for w in words):
            return True
    return False


def _indicqa_verbatim_pool(questions: list[str], rng: random.Random) -> list[dict]:
    """Dataset questions that survive the strict consumer-grade filter, verbatim
    (after suffix/prefix rescue-stripping)."""
    kept: list[str] = []
    for q in questions:
        q2 = _STRIP_PREFIX.sub("", _STRIP_SUFFIX.sub("?", q)).strip()
        q2 = re.sub(r"\s+\?$", "?", q2)
        if not q2.endswith("?"):
            q2 = q2.rstrip(".") + "?"
        if len(q2) < 25 or len(q2) > 200:
            continue
        if (_CASE_ANCHOR.search(q2) or _PARTY_WORDS.search(q2) or _PAST_RECALL.search(q2)
                or _PRACTITIONER.search(q2) or _PREDICTIVE.search(q2)
                or _DEICTIC_THIS.search(q2)):
            continue
        if not _LAW_SUBSTANCE.search(q2):
            continue
        if _has_party_name(q2):
            continue
        kept.append(q2)
    kept = sorted(set(kept))
    rng.shuffle(kept)
    return [make_seed(q, "en", [], keyword_domain(q), "indicqa") for q in kept]


# --- reference mining ------------------------------------------------------- #
_SECTION_REF = re.compile(
    r"[Ss]ections?\s+(\d+[A-Za-z\-]*(?:\(\d+\))?(?:\([a-z]\))?)\s*"
    r"(?:of\s+the\s+([A-Z][A-Za-z&,.'\- ]*?(?:Act|Code)(?:,?\s*\d{4})?)"
    r"|of\s+the\s+([A-Z][A-Za-z.]{1,8})\b"
    r"|\s+of\s+([A-Z][A-Za-z.]{1,8})\b"
    r"|\s+([A-Z][A-Za-z.]{1,8})\b)"
)
_ARTICLE_REF = re.compile(r"[Aa]rticles?\s+(\d+[A-Za-z\-]*)\s+of\s+the\s+Constitution")
_ACT_REF = re.compile(r"\bthe\s+([A-Z][A-Za-z&,.'\- ]*?(?:Act|Code|Sanhita)(?:,?\s*\d{4})?)(?=[\s,.?;)])")

ACT_CANON = {
    "ipc": "Indian Penal Code", "indian penal code": "Indian Penal Code",
    "penal code": "Indian Penal Code",
    "crpc": "Code of Criminal Procedure", "cr.p.c": "Code of Criminal Procedure",
    "cr.p.c.": "Code of Criminal Procedure", "crpc.": "Code of Criminal Procedure",
    "criminal procedure code": "Code of Criminal Procedure",
    "code of criminal procedure": "Code of Criminal Procedure",
    "cpc": "Code of Civil Procedure", "civil procedure code": "Code of Civil Procedure",
    "code of civil procedure": "Code of Civil Procedure",
    "ni act": "Negotiable Instruments Act, 1881",
    "negotiable instruments act": "Negotiable Instruments Act, 1881",
    "ndps act": "Narcotic Drugs and Psychotropic Substances Act, 1985",
    "pocso act": "Protection of Children from Sexual Offences Act, 2012",
    "pmla": "Prevention of Money Laundering Act, 2002",
    "uapa": "Unlawful Activities (Prevention) Act, 1967",
    "evidence act": "Indian Evidence Act, 1872",
    "indian evidence act": "Indian Evidence Act, 1872",
    "motor vehicles act": "Motor Vehicles Act, 1988", "mv act": "Motor Vehicles Act, 1988",
    "esi act": "Employees' State Insurance Act, 1948",
    "it act": "Information Technology Act, 2000",
    "information technology act": "Information Technology Act, 2000",
    "arbitration act": "Arbitration and Conciliation Act, 1996",
    "arbitration and conciliation act": "Arbitration and Conciliation Act, 1996",
    "hindu marriage act": "Hindu Marriage Act, 1955",
    "limitation act": "Limitation Act, 1963",
    "specific relief act": "Specific Relief Act, 1963",
    "land acquisition act": "Land Acquisition Act, 1894",
    "industrial disputes act": "Industrial Disputes Act, 1947",
    "income tax act": "Income Tax Act, 1961",
    "customs act": "Customs Act, 1962",
    "central excise act": "Central Excise Act, 1944",
    "consumer protection act": "Consumer Protection Act, 2019",
    "domestic violence act": "Protection of Women from Domestic Violence Act, 2005",
    "dv act": "Protection of Women from Domestic Violence Act, 2005",
    "dowry prohibition act": "Dowry Prohibition Act, 1961",
    "prevention of corruption act": "Prevention of Corruption Act, 1988",
    "companies act": "Companies Act, 2013",
    "contract act": "Indian Contract Act, 1872",
    "indian contract act": "Indian Contract Act, 1872",
    "sarfaesi act": "SARFAESI Act, 2002",
    "ibc": "Insolvency and Bankruptcy Code, 2016",
    "insolvency and bankruptcy code": "Insolvency and Bankruptcy Code, 2016",
    "juvenile justice act": "Juvenile Justice (Care and Protection of Children) Act, 2015",
    "rti act": "Right to Information Act, 2005",
    "right to information act": "Right to Information Act, 2005",
    "transfer of property act": "Transfer of Property Act, 1882",
    "registration act": "Registration Act, 1908",
    "payment of gratuity act": "Payment of Gratuity Act, 1972",
    "sc/st act": "SC/ST (Prevention of Atrocities) Act, 1989",
    "atrocities act": "SC/ST (Prevention of Atrocities) Act, 1989",
}
PENAL_ACTS = {
    "Indian Penal Code", "Narcotic Drugs and Psychotropic Substances Act, 1985",
    "Protection of Children from Sexual Offences Act, 2012",
    "Unlawful Activities (Prevention) Act, 1967", "Prevention of Corruption Act, 1988",
    "Prevention of Money Laundering Act, 2002", "Dowry Prohibition Act, 1961",
    "Arms Act, 1959", "Negotiable Instruments Act, 1881",
    "SC/ST (Prevention of Atrocities) Act, 1989",
}
ACT_DOMAIN = {
    "Code of Criminal Procedure": "police_procedure",
    "Indian Evidence Act, 1872": "evidence",
    "Negotiable Instruments Act, 1881": "property_fraud",
    "Information Technology Act, 2000": "property_fraud",
    "Motor Vehicles Act, 1988": "accident_mva",
    "Consumer Protection Act, 2019": "consumer",
    "Industrial Disputes Act, 1947": "labour",
    "Employees' State Insurance Act, 1948": "labour",
    "Payment of Gratuity Act, 1972": "labour",
    "Hindu Marriage Act, 1955": "family_dowry_dv",
    "Protection of Women from Domestic Violence Act, 2005": "family_dowry_dv",
    "Dowry Prohibition Act, 1961": "family_dowry_dv",
    "Right to Information Act, 2005": "documents_rti",
    "Registration Act, 1908": "documents_rti",
    "Transfer of Property Act, 1882": "tenancy",
}
_ACT_FIRSTWORD_BLACKLIST = {"amendment", "said", "impugned", "new", "old", "the"}
_ACT_SANE = re.compile(r"^[A-Z][A-Za-z&().,'\- ]*(Act|Code)(, \d{4})?$")


def _canon_act(raw: str) -> str | None:
    a = " ".join(raw.strip().rstrip(",").split())
    key = a.lower().rstrip(".")
    if key in ACT_CANON:
        return ACT_CANON[key]
    # bare acronym (IPC, FSSA, TADA…) possibly followed by "Act"
    if re.fullmatch(r"[A-Z][A-Za-z.]{1,8}", a):
        return ACT_CANON.get(key, a if a.isupper() and 2 < len(a) <= 8 else None)
    if not _ACT_SANE.match(a):
        return None
    words = a.split()
    if len(words) > 9 or words[0].lower() in _ACT_FIRSTWORD_BLACKLIST:
        return None
    return a


def _ipc_section_domain(sec: str) -> str:
    m = re.match(r"(\d+)", sec)
    n = int(m.group(1)) if m else 0
    if sec.upper().startswith("498") or sec.upper() in ("304B", "304-B"):
        return "family_dowry_dv"
    if sec.upper() in ("304A", "304-A", "279"):
        return "accident_mva"
    if 299 <= n <= 377:
        return "violent_crime"
    if 378 <= n <= 462:
        return "property_fraud"
    return "other"


def _ref_domain(act: str, sec: str | None) -> str:
    if act == "Indian Penal Code" and sec:
        return _ipc_section_domain(sec)
    return ACT_DOMAIN.get(act, keyword_domain(act))


_SECTION_FORMS = [
    "What does Section {s} of the {a} deal with?",
    "Explain Section {s} of the {a} in simple language.",
    "When does Section {s} of the {a} apply?",
    "What are the key requirements of Section {s} of the {a}?",
]
_SECTION_FORM_PENAL = "What is the punishment under Section {s} of the {a}?"
_ARTICLE_FORMS = [
    "What does Article {n} of the Constitution of India provide?",
    "Explain Article {n} of the Constitution of India in simple language.",
    "What rights or powers flow from Article {n} of the Constitution of India?",
]
_ACT_FORMS = [
    "What is the {a} about and when does it apply?",
    "What are the most important provisions of the {a}?",
    "Who is protected by the {a} and what remedies does it provide?",
]


def _indicqa_mined_pool(questions: list[str]) -> list[dict]:
    """Canonical consumer questions generated from the legal references the dataset's
    own questions cite. Ordered: most-cited refs first, one form-pass at a time, so a
    quota cut still covers every reference before repeating forms."""
    sec_refs: Counter = Counter()
    art_refs: Counter = Counter()
    act_refs: Counter = Counter()
    for q in questions:
        for m in _SECTION_REF.finditer(q):
            act = _canon_act(next(g for g in m.groups()[1:] if g))
            if act:
                sec_refs[(m.group(1), act)] += 1
        for m in _ARTICLE_REF.finditer(q):
            art_refs[m.group(1)] += 1
        for m in _ACT_REF.finditer(q):
            act = _canon_act(m.group(1))
            if act:
                act_refs[act] += 1

    sec_sorted = sorted(sec_refs, key=lambda r: (-sec_refs[r], r[1], r[0]))
    art_sorted = sorted(art_refs, key=lambda a: (-art_refs[a], a))
    act_sorted = sorted(act_refs, key=lambda a: (-act_refs[a], a))

    out: list[dict] = []
    max_passes = max(len(_SECTION_FORMS) + 1, len(_ARTICLE_FORMS), len(_ACT_FORMS))
    for p in range(max_passes):
        for sec, act in sec_sorted:
            forms = _SECTION_FORMS + ([_SECTION_FORM_PENAL] if act in PENAL_ACTS else [])
            if p < len(forms):
                q = forms[p].format(s=sec, a=act)
                out.append(make_seed(q, "en", [], _ref_domain(act, sec), "indicqa"))
        for n in art_sorted:
            if p < len(_ARTICLE_FORMS):
                q = _ARTICLE_FORMS[p].format(n=n)
                out.append(make_seed(q, "en", [], "other", "indicqa"))
        for act in act_sorted:
            if p < len(_ACT_FORMS):
                q = _ACT_FORMS[p].format(a=act)
                out.append(make_seed(q, "en", [], _ref_domain(act, None), "indicqa"))
    return out


def build_indicqa_candidates(rng: random.Random) -> list[dict]:
    with INDICQA_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    questions = sorted({" ".join(d.get("question", "").split()) for d in data} - {""})
    verbatim = _indicqa_verbatim_pool(questions, rng)
    mined = _indicqa_mined_pool(questions)
    print(f"  indicqa pools: {len(verbatim)} verbatim survivors "
          f"(of {len(questions)} unique dataset questions), {len(mined)} mined candidates")
    return verbatim + mined


# --------------------------------------------------------------------------- #
# Shared slot pools (English)
# --------------------------------------------------------------------------- #
AMOUNTS = ["8,000", "25,000", "60,000", "1.5 lakh", "3 lakh"]
TIMEFRAMES = ["last week", "two weeks ago", "a month ago", "three months ago"]
RELATIONS = ["brother", "father", "son", "uncle"]
CITIES = ["Lucknow", "Pune", "Patna", "Indore", "Jaipur"]

# --------------------------------------------------------------------------- #
# KIND 2 — narrative_en: 40 hand-authored lay-narrative templates (4 x 10 domains)
# --------------------------------------------------------------------------- #
EN_TEMPLATES: list[tuple[str, str, dict]] = [
    # ---- violent_crime ----
    ("violent_crime",
     "My {relation} was attacked with a {weapon} by {attackers} near our {place} {timeframe}. "
     "He is badly injured and in hospital. What case can be filed against them?",
     {"relation": RELATIONS, "weapon": ["steel rod", "knife", "cricket bat"],
      "attackers": ["three men from the neighbourhood", "a group of local goons",
                    "two men he had a money dispute with"],
      "place": ["house", "shop", "village"], "timeframe": ["last night", "two days ago"]}),
    ("violent_crime",
     "A man from our locality has been threatening to kill me over a {dispute} for the past "
     "{timeframe}. I have {evidence}. What offence is this and how do I get police protection?",
     {"dispute": ["land dispute", "parking dispute", "loan he has not repaid", "long-running rivalry"],
      "timeframe": ["two weeks", "one month", "three months"],
      "evidence": ["recordings of his calls", "witnesses from the market", "his threatening messages"]}),
    ("violent_crime",
     "Someone threw {substance} at my {relation} after she rejected his proposal. Her face is "
     "burnt and the doctors say she needs surgery. What does the law say and what compensation can we get?",
     {"substance": ["acid", "a corrosive chemical"],
      "relation": ["sister", "daughter", "cousin", "friend"]}),
    ("violent_crime",
     "My {relation} has been missing since {timeframe} and we suspect a man from the next "
     "village lured her away. The police keep telling us to wait. Is this kidnapping and what are our options?",
     {"relation": ["16-year-old daughter", "younger sister", "niece"],
      "timeframe": ["yesterday morning", "two days", "last Friday"]}),
    # ---- property_fraud (incl. cyber) ----
    ("property_fraud",
     "I paid Rs {amount} in advance to a seller on {platform} for a {object}, but he blocked "
     "my number and never delivered. What legal action can I take?",
     {"amount": AMOUNTS, "platform": ["Instagram", "OLX", "Facebook Marketplace", "a WhatsApp group"],
      "object": ["second-hand phone", "laptop", "camera", "bike"]}),
    ("property_fraud",
     "Someone called me pretending to be from my bank, got the OTP, and transferred Rs "
     "{amount} out of my account {timeframe}. Which law applies and where should I complain first?",
     {"amount": AMOUNTS, "timeframe": ["yesterday", "this morning", "last week"]}),
    ("property_fraud",
     "My {relation} gave Rs {amount} to a man who promised him {promise}, and now the man "
     "denies everything. Is this cheating and what proof do we need?",
     {"relation": ["father", "uncle", "friend"], "amount": AMOUNTS,
      "promise": ["a government job", "a plot of land", "a visa for Dubai", "a franchise dealership"]}),
    ("property_fraud",
     "Our house was broken into while we were away {timeframe} and {items} were stolen. "
     "Which sections apply and what should the FIR mention?",
     {"timeframe": ["for a wedding", "for two weeks", "over the weekend"],
      "items": ["gold jewellery and cash", "a laptop and important documents", "my wife's jewellery"]}),
    # ---- family_dowry_dv ----
    ("family_dowry_dv",
     "My in-laws keep demanding a {dowry_item} and Rs {amount} even after {years} years of "
     "marriage, and my husband beats me whenever his mother instigates him. What protection does the law give me?",
     {"dowry_item": ["car", "motorcycle", "flat"], "amount": ["2 lakh", "5 lakh", "50,000"],
      "years": ["two", "three", "five"]}),
    ("family_dowry_dv",
     "My husband left me at my parents' home {timeframe} ago and refuses to pay anything for "
     "me and our {children}. How do I claim maintenance from him?",
     {"timeframe": ["six months", "a year", "three months"],
      "children": ["two children", "infant daughter", "school-going son"]}),
    ("family_dowry_dv",
     "My wife and I have agreed to separate peacefully after {years} years of marriage. What "
     "is the procedure for a mutual-consent divorce and how long does it take?",
     {"years": ["four", "seven", "ten"]}),
    ("family_dowry_dv",
     "My sister died at her in-laws' house within {years} years of her marriage and they used "
     "to demand dowry constantly. The police are calling it suicide. What does the law say about such deaths?",
     {"years": ["two", "three", "four"]}),
    # ---- police_procedure ----
    ("police_procedure",
     "The police station in {city} refused to register my FIR about {complaint}. The officer "
     "told me to settle the matter myself. What can I do to get the FIR registered?",
     {"city": CITIES, "complaint": ["my stolen scooter", "a fraud of Rs 40,000", "an assault on my brother"]}),
    ("police_procedure",
     "My {relation} was picked up by the police {timeframe} ago and has still not been "
     "produced before a magistrate. What are his rights and what can our family do?",
     {"relation": ["son", "brother", "husband"], "timeframe": ["two days", "36 hours", "yesterday"]}),
    ("police_procedure",
     "I fear my {opponent} will file a false case against me over {dispute}. Can I apply for "
     "anticipatory bail before any arrest, and how does it work?",
     {"opponent": ["business partner", "neighbour", "wife's family"],
      "dispute": ["a property dispute", "a money dispute", "our ongoing divorce"]}),
    ("police_procedure",
     "An FIR was registered against my {relation} {timeframe} ago for a non-bailable offence. "
     "When can he apply for bail and what do courts consider before granting it?",
     {"relation": ["brother", "friend", "son"], "timeframe": TIMEFRAMES}),
    # ---- consumer ----
    ("consumer",
     "I bought a {appliance} for Rs {amount} from a shop in {city} and it stopped working "
     "within {timeframe}. The shop refuses to repair or refund even though it is under "
     "warranty. Can I take them to the consumer commission?",
     {"appliance": ["refrigerator", "washing machine", "smart TV", "water purifier"],
      "amount": ["18,000", "32,000", "55,000"], "city": CITIES,
      "timeframe": ["a month", "two weeks", "ten days"]}),
    ("consumer",
     "An e-commerce site delivered {wrong_thing} instead of the {object} I ordered for Rs "
     "{amount}, and customer care keeps closing my complaint without solving it. What are my rights?",
     {"wrong_thing": ["a fake product", "a damaged piece", "an empty box"],
      "object": ["mobile phone", "mixer grinder", "pair of shoes"], "amount": AMOUNTS}),
    ("consumer",
     "A {service_provider} took Rs {amount} as advance and then {failure}. Is this a "
     "deficiency in service and how much compensation can I claim?",
     {"service_provider": ["builder", "travel agency", "coaching institute", "wedding planner"],
      "amount": ["50,000", "2 lakh", "5 lakh"],
      "failure": ["kept delaying for months", "cancelled without refund", "shut down midway"]}),
    ("consumer",
     "A hospital in {city} charged me Rs {amount} but {negligence}. Can I file a medical "
     "negligence case in the consumer forum, and what should I collect as proof?",
     {"city": CITIES, "amount": ["80,000", "2.5 lakh", "4 lakh"],
      "negligence": ["the operation made my condition worse", "they left the treatment incomplete",
                     "a wrong injection caused a severe reaction"]}),
    # ---- labour ----
    ("labour",
     "I work in a {business} in {city}. My employer has not paid my salary of Rs {amount} per "
     "month for {timeframe}. Where do I complain and what does the law say about delayed wages?",
     {"business": ["garment factory", "private school", "security agency", "restaurant"],
      "city": CITIES, "amount": ["12,000", "18,000", "25,000"],
      "timeframe": ["two months", "three months", "four months"]}),
    ("labour",
     "My company terminated me {timeframe} ago without any notice or compensation after "
     "{years} years of service. What are my rights under the new labour codes?",
     {"timeframe": ["a month", "two weeks", "three months"], "years": ["three", "six", "nine"]}),
    ("labour",
     "My employer pays less than the minimum wage and makes us work {hours} hours a day with "
     "no overtime pay. Which law covers this and how do I complain without losing my job?",
     {"hours": ["12", "14", "11"]}),
    ("labour",
     "I resigned from my job {timeframe} ago but the company has not paid my final settlement "
     "and is withholding Rs {amount}. What legal remedy do I have?",
     {"timeframe": ["two months", "45 days", "four months"], "amount": ["40,000", "90,000", "1.2 lakh"]}),
    # ---- tenancy ----
    ("tenancy",
     "My landlord in {city} refuses to return my security deposit of Rs {amount} though I "
     "vacated the flat {timeframe} ago and have photos showing no damage. What can I do?",
     {"city": CITIES, "amount": ["30,000", "50,000", "1 lakh"],
      "timeframe": ["a month", "two months", "six weeks"]}),
    ("tenancy",
     "My landlord is demanding that I vacate within {timeframe} without any written notice "
     "because he says his son needs the flat. My rent agreement runs till {month}. What are my rights as a tenant?",
     {"timeframe": ["a week", "15 days", "one month"], "month": ["December", "next June", "March"]}),
    ("tenancy",
     "To force me out because I asked for rent receipts, the landlord has {harassment}. Is "
     "this legal, and how do I protect myself as a tenant?",
     {"harassment": ["cut off the water supply", "changed the main gate lock",
                     "started loud construction to make the flat unlivable"]}),
    ("tenancy",
     "I rented out my flat in {city} to a tenant who has not paid rent for {timeframe} and "
     "refuses to leave. What is the lawful way to evict him?",
     {"city": CITIES, "timeframe": ["four months", "six months", "nearly a year"]}),
    # ---- documents_rti ----
    ("documents_rti",
     "I applied for a {document} {timeframe} ago and the office keeps sending me back for one "
     "paper or another. Can I use RTI to ask about the status and the reason for the delay?",
     {"document": ["caste certificate", "domicile certificate", "birth certificate", "ration card"],
      "timeframe": ["two months", "six weeks", "four months"]}),
    ("documents_rti",
     "I filed an RTI application with a government department {timeframe} ago and have "
     "received no reply. What is the time limit for a response and how do I file a first appeal?",
     {"timeframe": ["forty days", "two months", "three months"]}),
    ("documents_rti",
     "The sub-registrar's office is delaying the registration of my {deed} without giving any "
     "reason. What are the rules for registration and can RTI help me here?",
     {"deed": ["sale deed", "gift deed", "partition deed"]}),
    ("documents_rti",
     "My father passed away leaving a {asset} but no will. What documents like a succession "
     "certificate or legal heir certificate do we need, and how do we get them?",
     {"asset": ["bank fixed deposit", "flat in the city", "piece of agricultural land"]}),
    # ---- evidence ----
    ("evidence",
     "The only proof I have against {person} is WhatsApp chats and call recordings. Are these "
     "admissible as evidence in an Indian court, and is any certificate needed?",
     {"person": ["the man who cheated me", "my business partner", "my tenant"]}),
    ("evidence",
     "The {incident} was captured on a nearby CCTV camera. How do I make sure the police "
     "seize and preserve the footage, and will it be accepted in court?",
     {"incident": ["assault on my brother", "theft outside my shop", "accident at the crossing"]}),
    ("evidence",
     "The main witness in our case is my {relation}. The other side says a family member's "
     "testimony does not count. Is that true under Indian evidence law?",
     {"relation": ["wife", "brother", "mother"]}),
    ("evidence",
     "Before dying in hospital, my {relation} told the doctor who had attacked him. Does such "
     "a dying statement count as evidence, and how much weight does it carry?",
     {"relation": ["uncle", "father", "neighbour"]}),
    # ---- accident_mva ----
    ("accident_mva",
     "My {relation} was hit by a speeding {vehicle} {timeframe} while crossing the road and "
     "the driver ran away. He has fractures and hospital bills of Rs {amount}. How do we claim compensation?",
     {"relation": RELATIONS, "vehicle": ["truck", "car", "tempo", "bus"],
      "timeframe": TIMEFRAMES, "amount": ["70,000", "1.5 lakh", "3 lakh"]}),
    ("accident_mva",
     "A {vehicle} jumped the red light and badly damaged my parked car; the repairs cost Rs "
     "{amount}. The driver admits fault but his insurance company is not responding. What are my options?",
     {"vehicle": ["delivery van", "jeep", "school bus"], "amount": ["45,000", "80,000", "1.2 lakh"]}),
    ("accident_mva",
     "My {relation} died in a road accident {timeframe} ago; he was earning Rs {amount} a "
     "month. How is compensation calculated for the family and where do we file the claim?",
     {"relation": ["father", "husband", "elder brother"], "timeframe": ["a month", "two months"],
      "amount": ["20,000", "35,000", "60,000"]}),
    ("accident_mva",
     "I was injured when a {vehicle} hit my bike; the police have charged the driver with "
     "rash driving. The insurer says I was partly at fault. Can I still claim compensation?",
     {"vehicle": ["car", "tempo", "pickup truck"]}),
]

# --------------------------------------------------------------------------- #
# KIND 3 — narrative_hi: 25 hand-written Devanagari templates (UTF-8 literals)
# --------------------------------------------------------------------------- #
HI_AMOUNTS = ["दस हज़ार", "पचास हज़ार", "एक लाख", "दो लाख", "पाँच लाख"]
HI_TIMEFRAMES = ["दो हफ़्ते", "एक महीने", "तीन महीने", "छह महीने", "कई दिनों"]
HI_RELATIONS_M = ["भाई", "पिता", "बेटे", "चाचा", "पति"]

HI_TEMPLATES: list[tuple[str, str, dict]] = [
    # ---- violent_crime (3) ----
    ("violent_crime",
     "मेरे {relation} पर {place} में कुछ लोगों ने {weapon} से हमला किया। वह अस्पताल में भर्ती है। हम उन लोगों पर क्या कानूनी कार्रवाई कर सकते हैं?",
     {"relation": HI_RELATIONS_M, "place": ["बाज़ार", "मोहल्ले", "खेत", "गली"],
      "weapon": ["लाठी", "चाकू", "लोहे की रॉड", "डंडे"]}),
    ("violent_crime",
     "पड़ोस का एक {attacker} मुझे {timeframe} से {reason} को लेकर जान से मारने की धमकी दे रहा है। मुझे बहुत डर लग रहा है, कानून में इसके लिए क्या है?",
     {"attacker": ["आदमी", "लड़का", "दबंग"], "timeframe": HI_TIMEFRAMES,
      "reason": ["ज़मीन के झगड़े", "पुराने पैसों", "पानी के विवाद", "गाड़ी खड़ी करने के झगड़े"]}),
    ("violent_crime",
     "मेरी बेटी का {place} आते-जाते समय एक लड़का {timeframe} से पीछा करता है और छेड़ता है। कानून में इसके लिए क्या सज़ा है और शिकायत कहां करें?",
     {"place": ["स्कूल", "कॉलेज", "ट्यूशन", "काम"],
      "timeframe": ["कई दिनों", "दो हफ़्तों", "एक महीने", "काफ़ी समय"]}),
    # ---- property_fraud (3) ----
    ("property_fraud",
     "मैंने ऑनलाइन {object} खरीदने के लिए {amount} रुपये दिए, लेकिन सामान नहीं आया और दुकानदार ने नंबर ब्लॉक कर दिया। अब मैं क्या करूं?",
     {"object": ["मोबाइल", "लैपटॉप", "साड़ी", "फर्नीचर", "सिलाई मशीन"], "amount": HI_AMOUNTS}),
    ("property_fraud",
     "किसी ने {method} मेरे बैंक खाते से {amount} रुपये निकाल लिए। यह कौन सा अपराध है और शिकायत कहां करें?",
     {"method": ["फोन पर ओटीपी पूछकर", "फर्जी लिंक भेजकर", "एटीएम कार्ड बदलकर", "बैंक अधिकारी बनकर"],
      "amount": HI_AMOUNTS}),
    ("property_fraud",
     "{timeframe} पहले मेरे घर से {object} चोरी हो गई। एफआईआर में कौन सी धारा लगेगी और पुलिस को क्या-क्या बताना चाहिए?",
     {"timeframe": ["दो दिन", "एक हफ़्ता", "तीन दिन", "कुछ दिन"],
      "object": ["सोने की चेन और नकदी", "मोटरसाइकिल", "अलमारी में रखी नकदी", "गहनों की पेटी"]}),
    # ---- family_dowry_dv (4) ----
    ("family_dowry_dv",
     "मेरे पति और सास {timeframe} से दहेज में {item} की मांग को लेकर मुझे ताने देते हैं और मारपीट करते हैं। मैं अपनी सुरक्षा के लिए क्या कर सकती हूं?",
     {"timeframe": HI_TIMEFRAMES, "item": ["गाड़ी", "मोटरसाइकिल", "नकद रकम", "ज़मीन"]}),
    ("family_dowry_dv",
     "मेरा पति {timeframe} से मुझे और {children} को खर्चा नहीं दे रहा और मायके भेज दिया है। क्या मैं गुज़ारा भत्ता मांग सकती हूं और कैसे?",
     {"timeframe": ["छह महीने", "एक साल", "तीन महीने", "आठ महीने"],
      "children": ["बच्चों", "छोटी बेटी", "स्कूल जाते बेटे"]}),
    ("family_dowry_dv",
     "मेरी शादी को {years} साल हुए हैं और पति रोज़ {behaviour} करता है। तलाक और बच्चों की कस्टडी के बारे में कानून क्या कहता है?",
     {"years": ["पांच", "आठ", "तीन", "दस"],
      "behaviour": ["शराब पीकर झगड़ा", "मारपीट", "गाली-गलौज", "शक करके परेशान"]}),
    ("family_dowry_dv",
     "मेरी बहन की ससुराल में शादी के {years} साल के अंदर संदिग्ध हालत में मौत हो गई। ससुराल वाले {demand} मांगते थे। कानून में ऐसे मामलों का क्या होता है?",
     {"years": ["दो", "तीन", "चार"], "demand": ["दहेज", "गाड़ी", "नकद पैसे", "ज़मीन का टुकड़ा"]}),
    # ---- police_procedure (3) ----
    ("police_procedure",
     "थाने में पुलिस मेरी {complaint} की एफआईआर नहीं लिख रही और कह रही है कि {excuse}। अब मेरे पास क्या रास्ता है?",
     {"complaint": ["चोरी", "मारपीट", "धोखाधड़ी", "गुमशुदगी"],
      "excuse": ["बाद में आना", "यह दीवानी मामला है", "पहले सबूत लाओ"]}),
    ("police_procedure",
     "मेरे {relation} को पुलिस {timeframe} पहले पकड़कर ले गई और अभी तक मजिस्ट्रेट के सामने पेश नहीं किया। हमारे क्या अधिकार हैं?",
     {"relation": HI_RELATIONS_M, "timeframe": ["दो दिन", "36 घंटे", "कल सुबह", "परसों"]}),
    ("police_procedure",
     "मुझ पर {opponent} की तरफ से {dispute} को लेकर झूठा केस होने का डर है। क्या मैं गिरफ्तारी से पहले अग्रिम ज़मानत ले सकता हूं? इसका तरीका क्या है?",
     {"opponent": ["पड़ोसी", "रिश्तेदारों", "बिज़नेस पार्टनर", "मकान मालिक", "गांव के दबंगों"],
      "dispute": ["ज़मीन", "पैसों के लेन-देन", "पारिवारिक झगड़े"]}),
    # ---- consumer (2) ----
    ("consumer",
     "मैंने {amount} रुपये में {appliance} खरीदा और वह {timeframe} में ही खराब हो गया। दुकानदार वारंटी के बाद भी पैसे वापस नहीं कर रहा। उपभोक्ता अदालत में शिकायत कैसे करूं?",
     {"amount": ["बीस हज़ार", "चालीस हज़ार", "साठ हज़ार"],
      "appliance": ["फ्रिज", "टीवी", "कूलर", "मिक्सर"],
      "timeframe": ["एक महीने", "दो हफ़्ते", "दस दिन", "चंद दिनों"]}),
    ("consumer",
     "ऑनलाइन कंपनी ने {object} की जगह {wrong} भेज दिया और अब रिफंड नहीं दे रही। उपभोक्ता कानून में मेरे क्या अधिकार हैं?",
     {"object": ["मोबाइल", "जूते", "मिक्सर", "प्रेशर कुकर"],
      "wrong": ["नकली सामान", "टूटा हुआ सामान", "खाली डिब्बा"]}),
    # ---- labour (3) ----
    ("labour",
     "मैं {city} की एक {business} में काम करता हूं। मालिक ने {timeframe} से तनख्वाह नहीं दी है। मैं कहां शिकायत करूं और कानून क्या कहता है?",
     {"city": ["दिल्ली", "कानपुर", "सूरत", "जयपुर", "पटना"], "business": ["फैक्टरी", "दुकान", "कंपनी"],
      "timeframe": ["दो महीने", "तीन महीने", "चार महीने"]}),
    ("labour",
     "मुझे {years} साल काम करने के बाद बिना नोटिस के नौकरी से निकाल दिया गया और {due} का भुगतान भी नहीं किया गया। नए श्रम कानूनों में मेरा क्या हक है?",
     {"years": ["तीन", "पांच", "आठ", "दस"],
      "due": ["आखिरी महीने की तनख्वाह", "कोई मुआवज़ा", "ग्रेच्युटी"]}),
    ("labour",
     "हमारा मालिक न्यूनतम मज़दूरी से कम पैसा देता है और रोज़ {hours} घंटे काम कराता है, महीने के सिर्फ {amount} रुपये देता है। इसकी शिकायत कहां और कैसे करें?",
     {"hours": ["बारह", "चौदह", "ग्यारह"], "amount": ["छह हज़ार", "आठ हज़ार", "नौ हज़ार"]}),
    # ---- tenancy (2) ----
    ("tenancy",
     "मकान मालिक मेरी {amount} रुपये की सिक्योरिटी जमा वापस नहीं कर रहा, जबकि मैंने {timeframe} पहले घर खाली कर दिया था। मैं क्या कर सकता हूं?",
     {"amount": ["तीस हज़ार", "पचास हज़ार", "एक लाख", "बीस हज़ार"],
      "timeframe": ["एक महीने", "दो महीने", "तीन हफ़्ते", "दो हफ़्ते"]}),
    ("tenancy",
     "मकान मालिक बिना लिखित नोटिस दिए {timeframe} में घर खाली करने को कह रहा है, जबकि किराए का एग्रीमेंट {period} तक का है। किराएदार के क्या अधिकार हैं?",
     {"timeframe": ["एक हफ़्ते", "पंद्रह दिन", "दस दिन"],
      "period": ["अगले साल मार्च", "दिसंबर", "अगले जून"]}),
    # ---- documents_rti (2) ----
    ("documents_rti",
     "मैंने {document} के लिए {timeframe} पहले आवेदन किया था, पर दफ्तर में कोई सुनवाई नहीं हो रही। क्या आरटीआई लगाकर देरी की वजह पूछ सकता हूं?",
     {"document": ["जाति प्रमाण पत्र", "राशन कार्ड", "जन्म प्रमाण पत्र", "आय प्रमाण पत्र"],
      "timeframe": ["दो महीने", "तीन महीने", "छह हफ़्ते", "चार महीने"]}),
    ("documents_rti",
     "{office} ने मेरी आरटीआई अर्ज़ी का {timeframe} से कोई जवाब नहीं दिया। जवाब की समय-सीमा क्या है और पहली अपील कैसे करूं?",
     {"office": ["तहसील कार्यालय", "नगर निगम", "बिजली विभाग", "शिक्षा विभाग"],
      "timeframe": ["चालीस दिन", "दो महीने", "तीन महीने"]}),
    # ---- evidence (2) ----
    ("evidence",
     "मेरे पास {person} के खिलाफ सबूत सिर्फ {evidence} है। क्या यह अदालत में सबूत माना जाएगा और इसके लिए क्या करना होगा?",
     {"person": ["धोखा देने वाले", "बिज़नेस पार्टनर", "किराएदार", "पड़ोसी"],
      "evidence": ["व्हाट्सएप चैट और कॉल रिकॉर्डिंग", "फोन की रिकॉर्डिंग", "मैसेज के स्क्रीनशॉट"]}),
    ("evidence",
     "{incident} की घटना पास के {camera} कैमरे में कैद हुई है। पुलिस से फुटेज कैसे सुरक्षित करवाएं और क्या वह अदालत में चलेगी?",
     {"incident": ["मारपीट", "चोरी", "एक्सीडेंट", "लूटपाट"],
      "camera": ["सीसीटीवी", "दुकान के सीसीटीवी", "सोसाइटी के सीसीटीवी"]}),
    # ---- accident_mva (2) ----
    ("accident_mva",
     "मेरे {relation} को {vehicle} ने टक्कर मार दी और ड्राइवर भाग गया। इलाज में {amount} रुपये लग चुके हैं। मुआवज़ा कैसे और कहां से मिलेगा?",
     {"relation": HI_RELATIONS_M, "vehicle": ["ट्रक", "कार", "टेम्पो"],
      "amount": ["पचास हज़ार", "एक लाख", "दो लाख"]}),
    ("accident_mva",
     "सड़क हादसे में मेरे {relation} की मौत हो गई, वे महीने के {amount} रुपये कमाते थे। परिवार को मुआवज़ा कैसे तय होता है और क्लेम कहां दाखिल करें?",
     {"relation": ["पिता", "पति", "बड़े भाई", "बेटे"],
      "amount": ["पंद्रह हज़ार", "पच्चीस हज़ार", "चालीस हज़ार"]}),
]

# --------------------------------------------------------------------------- #
# KIND 4 — followup: two-turn seeds (turn-1 narrative + STATIC assistant reply)
# --------------------------------------------------------------------------- #
# Each blueprint: (domain, language, turn1 template, slots, static assistant reply,
# list of turn-2 follow-up queries). The reply is 1-2 sentences naming the section
# for the template's known domain — synthesized statically, consistent with the
# project's IPC->BNS map (golden.jsonl expectations: 379->303, 420->318, 498A->85 …).
FOLLOWUP_BLUEPRINTS: list[dict] = [
    dict(domain="property_fraud", language="en",
         turn1="My {object} was stolen from {place} {timeframe}. What can I do?",
         slots={"object": ["phone", "scooter", "laptop", "gold chain"],
                "place": ["my office", "the market", "outside my house"],
                "timeframe": ["yesterday", "last night", "this week"]},
         reply="That would be theft under Section 303 of the Bharatiya Nyaya Sanhita, 2023 "
               "(earlier Section 379 of the IPC). You should report it at the nearest police "
               "station and get an FIR registered.",
         followups=["What is the punishment for that?",
                    "Can the thief get bail easily for this offence?",
                    "What happens after I file the FIR?",
                    "What if the police refuse to register my complaint?",
                    "Do I need a lawyer at the FIR stage?"]),
    dict(domain="property_fraud", language="en",
         turn1="I transferred Rs {amount} to an online seller for a {object} and he has "
               "blocked me everywhere. Is there any legal remedy?",
         slots={"amount": AMOUNTS, "object": ["phone", "watch", "concert ticket"]},
         reply="This is cheating under Section 318(4) of the Bharatiya Nyaya Sanhita, 2023 "
               "(earlier Section 420 of the IPC). Report it on the national cybercrime portal "
               "or helpline 1930, and also file a police complaint.",
         followups=["What is the punishment for that offence?",
                    "Will I get my money back through this process?",
                    "Is this a bailable offence?",
                    "How long do such cases usually take?"]),
    dict(domain="family_dowry_dv", language="en",
         turn1="My husband's family has been demanding Rs {amount} and a {item}, and my "
               "husband hit me during an argument {timeframe}. What are my options?",
         slots={"amount": ["2 lakh", "5 lakh"], "item": ["car", "motorcycle", "flat"],
                "timeframe": ["last week", "two days ago"]},
         reply="This is cruelty by a husband or his relatives under Section 85 of the "
               "Bharatiya Nyaya Sanhita, 2023 (earlier Section 498A of the IPC), and you can "
               "also seek protection orders under the Protection of Women from Domestic "
               "Violence Act, 2005.",
         followups=["What is the punishment for that?",
                    "Can I claim maintenance while living separately?",
                    "Will the police arrest them immediately after my complaint?",
                    "What proof should I keep before complaining?"]),
    dict(domain="violent_crime", language="en",
         turn1="{attacker} attacked my {relation} with a {weapon} and the doctors say the "
               "injuries are serious. What case can be filed?",
         slots={"attacker": ["A neighbour", "A group of men", "His business partner"],
                "relation": ["brother", "father", "cousin"],
                "weapon": ["knife", "hammer"]},
         reply="Depending on the intention and the injuries, this can be attempt to murder "
               "under Section 109 of the Bharatiya Nyaya Sanhita, 2023 (earlier Section 307 "
               "of the IPC) or voluntarily causing grievous hurt under Section 117.",
         followups=["What is the punishment for that?",
                    "Can the accused get bail for such an offence?",
                    "What should we tell the police for the FIR?",
                    "How long will the trial take?"]),
    dict(domain="police_procedure", language="en",
         turn1="The police at our local station refused to register my FIR about {complaint}. "
               "What should I do?",
         slots={"complaint": ["a theft in my shop", "an assault on my son",
                              "a fraud of Rs 60,000", "my missing daughter"]},
         reply="Registration of information about a cognizable offence is governed by Section "
               "173 of the Bharatiya Nagarik Suraksha Sanhita, 2023 (earlier Section 154 "
               "CrPC). If refused, you can send your complaint in writing to the "
               "Superintendent of Police, and after that approach a Magistrate.",
         followups=["What if the Superintendent of Police also does nothing?",
                    "Can I file the complaint online instead?",
                    "How long does the Magistrate route take?",
                    "Can the police be punished for refusing?"]),
    dict(domain="consumer", language="en",
         turn1="My new {appliance} worth Rs {amount} stopped working within {timeframe} and "
               "the seller refuses a refund or replacement. What can I do?",
         slots={"appliance": ["washing machine", "smart TV", "refrigerator"],
                "amount": ["22,000", "45,000", "70,000"],
                "timeframe": ["two weeks", "a month"]},
         reply="You can file a complaint before the District Consumer Disputes Redressal "
               "Commission under Section 35 of the Consumer Protection Act, 2019 for the "
               "defective product and deficiency in service.",
         followups=["How much is the fee for filing that complaint?",
                    "Can I claim compensation beyond the price of the product?",
                    "Do I need a lawyer in consumer court?",
                    "What is the time limit for filing the complaint?"]),
    dict(domain="labour", language="en",
         turn1="My employer has not paid my salary of Rs {amount} per month for {timeframe}. "
               "Which law protects me?",
         slots={"amount": ["12,000", "18,000", "25,000"],
                "timeframe": ["two months", "three months"]},
         reply="Timely payment of wages is required by Section 17 of the Code on Wages, 2019, "
               "and you can file a claim for the unpaid amount before the authority appointed "
               "under that Code.",
         followups=["What penalty does the employer face for this?",
                    "Where exactly do I file the claim?",
                    "Can he fire me for complaining?",
                    "What if I worked through a contractor agency?"]),
    dict(domain="accident_mva", language="en",
         turn1="A {vehicle} hit my {relation} and the driver fled the spot. The hospital "
               "bills are already Rs {amount}. How do we get compensation?",
         slots={"vehicle": ["truck", "car", "bus"], "relation": ["father", "brother"],
                "amount": ["60,000", "1.5 lakh"]},
         reply="Your family can file a compensation claim before the Motor Accident Claims "
               "Tribunal under Section 166 of the Motor Vehicles Act, 1988; for hit-and-run "
               "cases there is also a special compensation scheme under Section 161.",
         followups=["How is the compensation amount decided?",
                    "What if the vehicle is never traced?",
                    "Is there a time limit to file the claim?",
                    "Can we claim for future treatment costs too?"]),
    # ---- Hindi blueprints ----
    dict(domain="property_fraud", language="hi",
         turn1="मेरा {object} {place} से चोरी हो गया। मुझे क्या करना चाहिए?",
         slots={"object": ["मोबाइल", "स्कूटर", "पर्स"], "place": ["बाज़ार", "दफ्तर", "घर के बाहर"]},
         reply="यह भारतीय न्याय संहिता, 2023 की धारा 303 के तहत चोरी का अपराध है (पहले आईपीसी की धारा 379)। आप नज़दीकी थाने में एफआईआर दर्ज कराएं।",
         followups=["इसके लिए कितनी सज़ा हो सकती है?",
                    "क्या चोर को आसानी से ज़मानत मिल जाएगी?",
                    "एफआईआर के बाद आगे क्या होता है?",
                    "अगर पुलिस रिपोर्ट न लिखे तो क्या करूं?"]),
    dict(domain="family_dowry_dv", language="hi",
         turn1="ससुराल वाले {item} और {amount} रुपये की मांग कर रहे हैं और मेरा पति मारपीट करता है। मैं क्या करूं?",
         slots={"item": ["गाड़ी", "मोटरसाइकिल"], "amount": ["दो लाख", "पांच लाख"]},
         reply="यह भारतीय न्याय संहिता, 2023 की धारा 85 (पति या रिश्तेदारों द्वारा क्रूरता; पहले आईपीसी की धारा 498A) के तहत अपराध है। आप घरेलू हिंसा अधिनियम, 2005 के तहत भी संरक्षण मांग सकती हैं।",
         followups=["इसमें कितनी सज़ा होती है?",
                    "क्या मैं अलग रहकर गुज़ारा भत्ता मांग सकती हूं?",
                    "शिकायत के बाद क्या पुलिस उन्हें तुरंत गिरफ्तार करेगी?",
                    "मुझे कौन-कौन से सबूत रखने चाहिए?"]),
    dict(domain="property_fraud", language="hi",
         turn1="फोन पर ओटीपी पूछकर किसी ने मेरे खाते से {amount} रुपये निकाल लिए। यह कौन सा अपराध है?",
         slots={"amount": HI_AMOUNTS},
         reply="यह भारतीय न्याय संहिता, 2023 की धारा 318(4) के तहत ठगी (धोखाधड़ी) का अपराध है (पहले आईपीसी की धारा 420)। तुरंत साइबर हेल्पलाइन 1930 पर और cybercrime.gov.in पोर्टल पर शिकायत करें।",
         followups=["क्या मेरे पैसे वापस मिल सकते हैं?",
                    "इस अपराध में कितनी सज़ा है?",
                    "शिकायत के लिए कौन से कागज़ चाहिए?",
                    "क्या यह ज़मानती अपराध है?"]),
    dict(domain="police_procedure", language="hi",
         turn1="थाने में पुलिस मेरी {complaint} की रिपोर्ट नहीं लिख रही। अब क्या करूं?",
         slots={"complaint": ["चोरी", "मारपीट", "धोखाधड़ी"]},
         reply="संज्ञेय अपराध की सूचना दर्ज करना भारतीय नागरिक सुरक्षा संहिता, 2023 की धारा 173 (पहले सीआरपीसी की धारा 154) के तहत पुलिस का कर्तव्य है। मना करने पर आप पुलिस अधीक्षक को लिखित शिकायत भेज सकते हैं और फिर मजिस्ट्रेट के पास जा सकते हैं।",
         followups=["अगर पुलिस अधीक्षक भी न सुनें तो क्या करूं?",
                    "क्या ऑनलाइन शिकायत की जा सकती है?",
                    "मजिस्ट्रेट के पास जाने में कितना समय लगता है?"]),
]

# --------------------------------------------------------------------------- #
# KIND 5 — thin: legal-ish weak-context queries (Mode-B bait)
# --------------------------------------------------------------------------- #
THIN_STATES = [
    "Nagaland", "Manipur", "Tripura", "Sikkim", "Mizoram", "Meghalaya",
    "Arunachal Pradesh", "Goa", "Himachal Pradesh", "Uttarakhand", "Chhattisgarh",
    "Jharkhand", "Puducherry", "Ladakh", "the Andaman and Nicobar Islands", "Lakshadweep",
]
THIN_STATE_TEMPLATES: list[tuple[str, str]] = [
    ("documents_rti", "What is the stamp duty on a gift deed to a family member in {state}?"),
    ("tenancy", "What are the rent control rules for commercial shops in {state}?"),
    ("labour", "What is the current minimum wage for shop workers in {state}?"),
    ("other", "How much court fee is payable for a money recovery suit in {state}?"),
    ("documents_rti", "What is the procedure for mutation of agricultural land in {state}?"),
    ("tenancy", "Are there special police verification rules for tenants in {state}?"),
]
THIN_OBSCURE: list[tuple[str, str]] = [
    ("other", "What is the penalty for running an unregistered boiler under the Indian Boilers Act, 1923?"),
    ("other", "What does the Sarais Act, 1867 require from a hotel or dharamshala owner?"),
    ("other", "What fine applies under the Cattle Trespass Act, 1871 if someone's cattle enter my field?"),
    ("other", "What are my obligations under the Indian Treasure Trove Act, 1878 if I find old coins buried in my land?"),
    ("other", "Is a licence needed under the Poisons Act, 1919 to sell rat poison in a general store?"),
    ("other", "What does the Public Gambling Act, 1867 say about playing cards for money at home?"),
    ("documents_rti", "How can a death that happened 20 years ago be registered under the Registration of Births and Deaths Act?"),
    ("other", "Does the Indian Post Office Act punish sending prohibited articles by post?"),
    ("other", "What is the punishment under the Prevention of Insults to National Honour Act for disrespecting the national flag at a private event?"),
    ("other", "What are the licensing requirements under the Petroleum Act, 1934 for storing diesel for a generator?"),
    ("other", "Under the Places of Worship Act, 1991, can the character of a local shrine be changed?"),
    ("other", "What powers does the Epidemic Diseases Act, 1897 give to the district administration?"),
]
THIN_VAGUE: list[tuple[str, str]] = [
    ("other", "my neighbour keeps troubling me is there any law for this"),
    ("other", "can I take legal action against someone for lying about me to others"),
    ("other", "someone gave me a bad look and abused me in public, can I file a case"),
    ("consumer", "is it illegal if a shopkeeper refuses to sell me something"),
    ("property_fraud", "my friend is not returning my bike since two months, is that a crime"),
    ("other", "what are my rights if my society secretary is rude to me"),
    ("property_fraud", "can I go to police if someone blocks me everywhere after borrowing money"),
    ("other", "there is too much noise from a banquet hall near my house every night, which law applies"),
    ("other", "my relative did black magic on us, does the law punish this"),
    ("other", "a stray dog bit my son near our society, who is legally responsible"),
    ("consumer", "an astrologer took money and his prediction was wrong, can I get a refund legally"),
    ("labour", "my colleague keeps taking credit for my work, is there any legal remedy"),
    ("other", "is it against the law to not return wedding gift money when asked"),
    ("family_dowry_dv", "my grown-up son refuses to talk to us, can we take back the property we gave him"),
]
THIN_HINDI: list[tuple[str, str]] = [
    ("other", "हमारे गांव में पंचायत ने मुझ पर जुर्माना लगाया है, क्या यह कानूनी है?"),
    ("other", "पड़ोसी की भैंस रोज़ मेरे खेत में घुस जाती है, इसके लिए क्या कानून है?"),
    ("family_dowry_dv", "क्या शादी में मिले गहनों पर बहू का कानूनी हक होता है?"),
    ("documents_rti", "मंदिर के चंदे का हिसाब कोई नहीं देता, क्या कानून से पूछ सकते हैं?"),
    ("other", "रात में देर तक डीजे बजाने पर कौन सा कानून लगता है?"),
    ("other", "गांव का चौकीदार हर महीने वसूली करता है, इसकी शिकायत कहां करें?"),
]


# --------------------------------------------------------------------------- #
# Candidate builders (each returns a deterministic, interleaved candidate list)
# --------------------------------------------------------------------------- #
def build_narrative_candidates(templates: list[tuple[str, str, dict]], language: str,
                               kind: str, rng: random.Random) -> list[dict]:
    pools = []
    for domain, tmpl, slots in templates:
        variants = expand(tmpl, slots, rng)
        pools.append([make_seed(q, language, [], domain, kind) for q in variants])
    return round_robin(pools)


def build_followup_candidates(rng: random.Random) -> list[dict]:
    pools = []
    for bp in FOLLOWUP_BLUEPRINTS:
        turn1s = expand(bp["turn1"], bp["slots"], rng)
        pairs = [(t1, fu) for t1 in turn1s for fu in bp["followups"]]
        rng.shuffle(pairs)
        pool = []
        for t1, fu in pairs:
            history = [
                {"role": "user", "content": t1},
                {"role": "assistant", "content": bp["reply"]},
            ]
            pool.append(make_seed(fu, bp["language"], history, bp["domain"], "followup"))
        pools.append(pool)
    return round_robin(pools)


def build_thin_candidates(rng: random.Random) -> list[dict]:
    pools = []
    for domain, tmpl in THIN_STATE_TEMPLATES:
        variants = expand(tmpl, {"state": THIN_STATES}, rng)
        pools.append([make_seed(q, "en", [], domain, "thin") for q in variants])
    for fixed, lang in ((THIN_OBSCURE, "en"), (THIN_VAGUE, "en"), (THIN_HINDI, "hi")):
        fixed = list(fixed)
        rng.shuffle(fixed)
        pools.append([make_seed(q, lang, [], domain, "thin") for domain, q in fixed])
    return round_robin(pools)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def load_golden_norms(path: Path) -> list[str]:
    norms = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                norms.append(norm(json.loads(line)["query"]))
    return [n for n in norms if n]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build deterministic distillation seed queries.")
    ap.add_argument("--total", type=int, default=2400, help="total number of seeds (default 2400)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for deterministic output")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output JSONL path")
    ap.add_argument("--show-samples", type=int, default=0, metavar="N",
                    help="print N sample seeds per kind after building")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    golden_norms = load_golden_norms(GOLDEN_JSONL)
    print(f"Loaded {len(golden_norms)} golden eval queries for exclusion.")

    kind_quotas = quotas(args.total)
    print("Quotas:", ", ".join(f"{k}={v}" for k, v in kind_quotas.items()))

    # Candidate lists are built in a FIXED order (rng state is shared and sequential).
    candidates = {
        "indicqa": build_indicqa_candidates(rng),
        "narrative_en": build_narrative_candidates(EN_TEMPLATES, "en", "narrative_en", rng),
        "narrative_hi": build_narrative_candidates(HI_TEMPLATES, "hi", "narrative_hi", rng),
        "followup": build_followup_candidates(rng),
        "thin": build_thin_candidates(rng),
    }
    for k, c in candidates.items():
        print(f"  {k}: {len(c)} candidates available")

    selector = SeedSelector(golden_norms)
    chosen: dict[str, list[dict]] = {}
    leftovers: dict[str, list[dict]] = {}
    for kind, _ in MIX:
        chosen[kind], leftovers[kind] = selector.take(candidates[kind], kind_quotas[kind])

    # Shortfall redistribution (safety valve — quotas should normally be met).
    shortfall = args.total - sum(len(v) for v in chosen.values())
    if shortfall > 0:
        print(f"WARNING: {shortfall} seed(s) short after quotas — redistributing from leftovers.")
        for kind in ("narrative_en", "thin", "followup", "narrative_hi", "indicqa"):
            extra, leftovers[kind] = selector.take(leftovers[kind], shortfall)
            chosen[kind].extend(extra)
            shortfall -= len(extra)
            if shortfall <= 0:
                break
        if shortfall > 0:
            print(f"WARNING: still {shortfall} short — all candidate pools exhausted.")

    seeds = [s for kind, _ in MIX for s in chosen[kind]]
    assert len({s["id"] for s in seeds}) == len(seeds), "duplicate seed ids"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for s in seeds:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # ---- stats ----
    print(f"\nWrote {len(seeds)} seeds -> {out_path}")
    print(f"Excluded: {selector.excluded_golden} golden-like, {selector.excluded_dupe} duplicates.")
    by_kind = Counter(s["kind"] for s in seeds)
    print("\nmix by kind:")
    for kind, _ in MIX:
        n = by_kind.get(kind, 0)
        print(f"  {kind:<14} {n:>5}  ({n / max(len(seeds), 1):.1%})")
    by_lang = Counter(s["language"] for s in seeds)
    print("by language: " + ", ".join(f"{k}={v}" for k, v in sorted(by_lang.items())))
    by_domain = Counter(s["domain"] for s in seeds)
    print("by domain:")
    for dom, n in by_domain.most_common():
        print(f"  {dom:<18} {n:>5}")
    with_history = sum(1 for s in seeds if s["history"])
    print(f"seeds with history (two-turn): {with_history}")

    if args.show_samples > 0:
        for kind, _ in MIX:
            print(f"\n--- samples: {kind} ---")
            for s in chosen[kind][: args.show_samples]:
                print(f"  [{s['language']}|{s['domain']}] {s['query']}")
                if s["history"]:
                    print(f"      turn1(user): {s['history'][0]['content']}")
                    print(f"      turn1(asst): {s['history'][1]['content'][:120]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
