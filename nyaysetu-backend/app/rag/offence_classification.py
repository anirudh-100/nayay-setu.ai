"""Offence classification lookup — the BNSS First Schedule, as a safe per-section map.

The case analysis must never *guess* whether an offence is cognizable/bailable/which-court
(see the adversarial review): that's only in the First Schedule, not the section text. This
loads the parsed, verified table (data/procedure/bnss_first_schedule.csv, built by
scripts/ingest_bnss_first_schedule.py) and exposes a single classification per BNS section
— but ONLY for the sections the parser marked unambiguous (one agreed value across all its
sub-rows). Conditional offences (e.g. theft, where it depends on value) return None and are
left unstated rather than over-simplified.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_BASE = re.compile(r"(\d+[A-Za-z]?)")

# Plain Hindi for the canonical First-Schedule values (kept in English form in the data).
_COG_HI = {"Cognizable": "संज्ञेय", "Non-cognizable": "असंज्ञेय"}
_BAIL_HI = {"Bailable": "जमानती", "Non-bailable": "गैर-जमानती"}
_COURT_HI = {
    "Any Magistrate": "कोई भी मजिस्ट्रेट",
    "Magistrate of the first class": "प्रथम श्रेणी मजिस्ट्रेट",
    "Magistrate of the second class": "द्वितीय श्रेणी मजिस्ट्रेट",
    "Court of Session": "सत्र न्यायालय",
}


def _base(section: str) -> str:
    m = _BASE.search(section or "")
    return m.group(1).upper() if m else ""


class OffenceClassification:
    """Singleton map: BNS base section -> {cognizable, bailable, court} (unambiguous only)."""

    _instance: Optional["OffenceClassification"] = None

    def __init__(self) -> None:
        self._by_section: dict[str, dict] = {}
        path = Path(settings.data_dir) / "procedure" / "bnss_first_schedule.csv"
        if not path.exists():
            logger.warning("BNSS First Schedule not found at %s — classification disabled.", path)
            return
        try:
            with path.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if (row.get("unambiguous") or "").strip().lower() != "yes":
                        continue
                    cog, bail = (row.get("cognizable") or "").strip(), (row.get("bailable") or "").strip()
                    if not (cog and bail):
                        continue
                    self._by_section[_base(row.get("section", ""))] = {
                        "cognizable": cog,
                        "bailable": bail,
                        "court": (row.get("court") or "").strip(),
                    }
            logger.info("Loaded %d unambiguous offence classifications (BNSS First Schedule)", len(self._by_section))
        except Exception as e:  # never let a bad CSV break startup
            logger.warning("Failed to load offence classifications: %s", e)
            self._by_section = {}

    @classmethod
    def instance(cls) -> "OffenceClassification":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def classify(self, section: str) -> Optional[dict]:
        """The unambiguous classification for a BNS base section, or None."""
        return self._by_section.get(_base(section))

    def describe(self, section: str, hindi: bool = False) -> Optional[str]:
        """One grounded sentence ('This is a cognizable, non-bailable offence, triable by
        the Court of Session.'), or None if the section has no unambiguous classification."""
        c = self.classify(section)
        if not c:
            return None
        cog, bail, court = c["cognizable"], c["bailable"], c.get("court", "")
        if hindi:
            parts = f"{_COG_HI.get(cog, cog)}, {_BAIL_HI.get(bail, bail)}"
            s = f"यह एक {parts} अपराध है"
            if court:
                s += f", जिसकी सुनवाई {_COURT_HI.get(court, court)} द्वारा होती है"
            return s + "। (स्रोत: BNSS पहली अनुसूची)"
        s = f"This is a {cog.lower()}, {bail.lower()} offence"
        if court:
            s += f", triable by {'the ' if court.startswith('Court') else ''}{court}"
        return s + ". (Source: BNSS First Schedule.)"
