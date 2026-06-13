"""Date-aware repealed->current law mapping — the correctness layer that keeps the engine current.

Since 1 July 2024 three new codes replaced their colonial-era predecessors:

    IPC  (1860)  ->  BNS   (Bharatiya Nyaya Sanhita, 2023)        — offences
    CrPC (1973)  ->  BNSS  (Bharatiya Nagarik Suraksha Sanhita)   — procedure
    IEA  (1872)  ->  BSA   (Bharatiya Sakshya Adhiniyam, 2023)    — evidence

Both old and new matter: new matters are charged/tried under the new codes, but anything
before the transition stays under the old ones. A credible tool must (a) point users to the
**current** section when they cite an old one, and (b) still answer about the old code for
historic matters. This service is the single source of truth for all three bridges.

It loads every mapping file in ``data/mappings/`` (each declares its ``from_code`` /
``to_code`` in ``_meta``), so adding a new transition is a data drop — no code change. The
mappings are flagged unverified; callers present them as guidance to confirm, never as the
authoritative bare act.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from threading import Lock
from typing import Optional

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Day the new criminal codes came into force (default when a map omits its own date).
TRANSITION_DATE = date(2024, 7, 1)

_BASE_SECTION_RE = re.compile(r"^\s*(\d+[A-Za-z]?)")


def _base_section(section: str) -> str:
    """'318(4)' -> '318', '326A' -> '326A'. Used to match across subsections."""
    m = _BASE_SECTION_RE.match(section or "")
    return m.group(1) if m else (section or "").strip()


def _parse_date(value: Optional[str]) -> date:
    if not value:
        return TRANSITION_DATE
    try:
        return date.fromisoformat(value)
    except ValueError:
        return TRANSITION_DATE


def _format_date(d: date) -> str:
    """'1 July 2024' — built manually because strftime('%-d') is not portable to Windows."""
    return f"{d.day} {d.strftime('%B')} {d.year}"


@dataclass
class _CodeMap:
    """One repealed->current correspondence table (e.g. IPC->BNS)."""

    from_code: str
    to_code: str
    transition_date: date
    verified: bool
    # base old section -> {"new": "318(4)"|None, "offence": str, "note": str}
    forward: dict[str, dict] = field(default_factory=dict)
    # base new section -> [old sections]
    reverse: dict[str, list[str]] = field(default_factory=dict)


class LawMap:
    """Singleton over every curated repealed<->current correspondence table."""

    _instance: "LawMap | None" = None
    _lock = Lock()

    def __init__(self, mappings_dir: Path | None = None) -> None:
        mappings_dir = mappings_dir or (Path(settings.data_dir) / "mappings")
        self._maps: dict[str, _CodeMap] = {}  # keyed by from_code (upper)

        if not mappings_dir.exists():
            logger.warning("Mappings dir not found at %s; running without current-law maps.", mappings_dir)
            return

        for path in sorted(mappings_dir.glob("*.json")):
            try:
                self._load_one(path)
            except Exception:
                logger.exception("Failed to load mapping file %s", path)

        if self._maps:
            logger.info(
                "Loaded %d current-law map(s): %s",
                len(self._maps),
                ", ".join(f"{m.from_code}->{m.to_code}({len(m.forward)})" for m in self._maps.values()),
            )

    def _load_one(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data.get("_meta", {})
        from_code = meta.get("from_code")
        to_code = meta.get("to_code")
        if not from_code or not to_code:
            # Not a transition table (e.g. a README/registry) — skip quietly.
            return

        cmap = _CodeMap(
            from_code=from_code,
            to_code=to_code,
            transition_date=_parse_date(meta.get("transition_date")),
            verified=bool(meta.get("verified", False)),
        )

        # The correspondence dict lives under "<from>_to_<to>" (e.g. "ipc_to_bns").
        data_key = f"{from_code.lower()}_to_{to_code.lower()}"
        table = data.get(data_key, {})
        for old_sec, entry in table.items():
            # The new-section value is keyed by the lowercased to_code (e.g. "bns", "bnss").
            new_sec = entry.get(to_code.lower())
            cmap.forward[_base_section(old_sec)] = {
                "new": new_sec,
                "offence": entry.get("offence", ""),
                "note": entry.get("note", ""),
            }
            if new_sec:
                cmap.reverse.setdefault(_base_section(new_sec), []).append(old_sec)

        self._maps[from_code.upper()] = cmap

    # ------------------------------------------------------------------ #
    @classmethod
    def instance(cls) -> "LawMap":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None

    # ------------------------------------------------------------------ #
    # Generic lookups (work for any loaded transition)
    # ------------------------------------------------------------------ #
    def from_codes(self) -> list[str]:
        """Repealed codes that have a successor map, e.g. ['IPC', 'CrPC', 'IEA']."""
        return [m.from_code for m in self._maps.values()]

    def successor(self, from_code: str, section: str) -> Optional[dict]:
        """Return {new, to_code, offence, note} for an old section, or None.

        ``new`` may be ``None`` for offences/provisions the new code did not retain."""
        cmap = self._maps.get((from_code or "").upper())
        if cmap is None:
            return None
        entry = cmap.forward.get(_base_section(section))
        if entry is None:
            return None
        return {**entry, "to_code": cmap.to_code}

    def predecessors(self, to_code: str, section: str) -> list[str]:
        """Old section(s) a current section corresponds to (may be empty)."""
        base = _base_section(section)
        out: list[str] = []
        for cmap in self._maps.values():
            if cmap.to_code.upper() == (to_code or "").upper():
                out.extend(cmap.reverse.get(base, []))
        return out

    def verified_for(self, from_code: str) -> bool:
        cmap = self._maps.get((from_code or "").upper())
        return bool(cmap and cmap.verified)

    def transition_date_for(self, from_code: str) -> date:
        cmap = self._maps.get((from_code or "").upper())
        return cmap.transition_date if cmap else TRANSITION_DATE

    def applicable_code(self, incident: Optional[date], *, from_code: str = "IPC") -> str:
        """Which code governs an incident on ``incident`` (the new code, the old, or 'either')."""
        cmap = self._maps.get((from_code or "").upper())
        if cmap is None:
            return "either"
        if incident is None:
            return "either"
        return cmap.to_code if incident >= cmap.transition_date else cmap.from_code

    # ------------------------------------------------------------------ #
    # Presentation helper
    # ------------------------------------------------------------------ #
    def current_reference_note(self, from_code: str, section: str) -> Optional[str]:
        """Human note bridging an old section to its current equivalent."""
        cmap = self._maps.get((from_code or "").upper())
        if cmap is None:
            return None
        entry = cmap.forward.get(_base_section(section))
        if entry is None:
            return None
        if entry.get("new") is None:
            note = entry.get("note", "")
            return (
                f"{from_code} Section {section} ({entry.get('offence', '')}) has no direct "
                f"equivalent in the {cmap.to_code}. {note}".strip()
            )
        suffix = f" Note: {entry['note']}" if entry.get("note") else ""
        return (
            f"{from_code} Section {section} now corresponds to {cmap.to_code} Section "
            f"{entry['new']} for matters on or after {_format_date(cmap.transition_date)}.{suffix}"
        )

    # ------------------------------------------------------------------ #
    # Backward-compatible IPC<->BNS helpers (kept so existing callers/tests work)
    # ------------------------------------------------------------------ #
    @property
    def verified(self) -> bool:
        """Verified flag of the IPC->BNS map (legacy accessor)."""
        return self.verified_for("IPC")

    def bns_for_ipc(self, ipc_section: str) -> Optional[dict]:
        """Legacy shape: {bns, offence, note} for an IPC section, or None."""
        entry = self.successor("IPC", ipc_section)
        if entry is None:
            return None
        return {"bns": entry["new"], "offence": entry.get("offence"), "note": entry.get("note")}

    def ipc_for_bns(self, bns_section: str) -> list[str]:
        """Legacy: IPC section(s) a BNS section corresponds to (may be empty)."""
        return self.predecessors("BNS", bns_section)
