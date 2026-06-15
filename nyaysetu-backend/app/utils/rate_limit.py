"""Tiny in-memory per-client rate limiter (fixed 60s window).

A public legal endpoint that calls a paid LLM needs a basic abuse/cost guard so a
single client can't run up the bill. This is intentionally minimal — per-IP counts
in memory, no external store — which is enough for our single-box deploy. For a
multi-instance setup, swap in a shared store (e.g. Redis); the interface stays the same.
"""
from __future__ import annotations

import time
from collections import deque
from threading import Lock


class RateLimiter:
    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = {}
        self._lock = Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        """True if ``key`` may make another request now. Disabled (always True) when
        per_minute <= 0. ``now`` is injectable for testing."""
        if self.per_minute <= 0:
            return True
        now = time.monotonic() if now is None else now
        window_start = now - 60.0
        with self._lock:
            dq = self._hits.setdefault(key, deque())
            while dq and dq[0] < window_start:
                dq.popleft()
            if len(dq) >= self.per_minute:
                return False
            dq.append(now)
            return True
