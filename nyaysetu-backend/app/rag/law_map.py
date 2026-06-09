"""Date-aware IPC <-> BNS mapping — the correctness layer that makes the engine current.

Since 1 July 2024 the Bharatiya Nyaya Sanhita (BNS) replaced the Indian Penal Code.
Both still matter: new offences are charged under the BNS, but offences *before* that
date remain under the IPC. A credible legal tool must therefore (a) point users to the
**current** section when they ask about an IPC section, and (b) still answer about the
IPC for historic matters. This service is the single source of truth for that bridge.

It loads the curated mapping (``data/mappings/ipc_bns.json``) once and exposes cheap
lookups in both directions, plus helpers to build the "IPC 420 now corresponds to BNS
Section 318" note the answer surfaces. The mapping is flagged unverified — callers
should present it as guidance to confirm, never as the authoritative bare act.
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from threading import Lock
from typing import Optional

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Day the new criminal codes came into force.
TRANSITION_DATE = date(2024, 7, 1)

_BASE_SECTION_RE = re.compile(r"^\s*(\d+[A-Za-z]?)")


def _base_section(section: str) -> str:
    """'318(4)' -> '318', '326A' -> '326A'. Used to match across subsections."""
    m = _BASE_SECTION_RE.match(section or "")
    return m.group(1) if m else (section or "").strip()


class LawMap:
    """Singleton over the curated IPC<->BNS correspondence table."""

    _instance: "LawMap | None" = None
    _lock = Lock()

    def __init__(self, path: Path | None = None) -> None:
        path = path or (Path(settings.data_dir) / "mappings" / "ipc_bns.json")
        self._verified = False
        self._ipc_to_bns: dict[str, dict] = {}
        self._bns_to_ipc: dict[str, list[str]] = {}

        if not path.exists():
            logger.warning("IPC<->BNS mapping not found at %s; running without it.", path)
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        self._verified = bool(data.get("_meta", {}).get("verified", False))
        self._ipc_to_bns = data.get("ipc_to_bns", {})

        # Build the reverse index (BNS base section -> list of IPC sections).
        for ipc_sec, entry in self._ipc_to_bns.items():
            bns = entry.get("bns")
            if not bns:
                continue
            base = _base_section(bns)
            self._bns_to_ipc.setdefault(base, []).append(ipc_sec)

        logger.info(
            "Loaded IPC<->BNS map: %d IPC entries (verified=%s)",
            len(self._ipc_to_bns),
            self._verified,
        )

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

    @property
    def verified(self) -> bool:
        return self._verified

    # ------------------------------------------------------------------ #
    # Lookups
    # ------------------------------------------------------------------ #
    def bns_for_ipc(self, ipc_section: str) -> Optional[dict]:
        """Return {bns, offence, note?} for an IPC section, or None.

        ``bns`` may be ``None`` for offences the BNS did not retain (e.g. 377)."""
        return self._ipc_to_bns.get(_base_section(ipc_section))

    def ipc_for_bns(self, bns_section: str) -> list[str]:
        """Return the IPC section(s) a BNS section corresponds to (may be empty)."""
        return self._bns_to_ipc.get(_base_section(bns_section), [])

    # ------------------------------------------------------------------ #
    # Presentation helpers
    # ------------------------------------------------------------------ #
    def current_reference_note(self, ipc_section: str) -> Optional[str]:
        """Human note bridging an IPC section to its current BNS equivalent."""
        entry = self.bns_for_ipc(ipc_section)
        if entry is None:
            return None
        if entry.get("bns") is None:
            note = entry.get("note", "")
            return (
                f"IPC Section {ipc_section} ({entry.get('offence','')}) has no direct "
                f"equivalent in the BNS. {note}".strip()
            )
        suffix = f" Note: {entry['note']}" if entry.get("note") else ""
        return (
            f"IPC Section {ipc_section} now corresponds to BNS Section {entry['bns']} "
            f"for offences on or after 1 July 2024.{suffix}"
        )

    def applicable_code(self, incident: Optional[date]) -> str:
        """Which code governs an incident on ``incident`` ('BNS', 'IPC', or 'either')."""
        if incident is None:
            return "either"
        return "BNS" if incident >= TRANSITION_DATE else "IPC"
