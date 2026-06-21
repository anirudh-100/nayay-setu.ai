"""Offline smoke test for the deployment pieces — rate limiter + index bootstrap.

Validates the production-only code paths without a real host:
  - RateLimiter: enforces the per-key window, resets after 60s, is per-key, and is a
    no-op when disabled (per_minute <= 0),
  - index_bootstrap.ensure_index(): downloads a zip over HTTP and reconstructs the
    expected index layout (qdrant/ + bm25.pkl), is idempotent, and fails safely.

Uses a throwaway HTTP server + temp dirs, so the real 105 MB index is never touched.

Usage:
    python scripts/deploy_smoke_test.py
"""
from __future__ import annotations

import functools
import http.server
import shutil
import tempfile
import threading
import zipfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.config import settings  # noqa: E402
from app.utils.index_bootstrap import ensure_index  # noqa: E402
from app.utils.rate_limit import RateLimiter  # noqa: E402

_failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global _failures
    if not ok:
        _failures += 1
    line = f"  [{'PASS' if ok else 'FAIL'}] {name}"
    if detail:
        line += f"  - {detail}"
    print(line)


def test_rate_limiter() -> None:
    print("\nRate limiter:")
    rl = RateLimiter(2)
    check("1st request allowed", rl.allow("ip", now=100.0))
    check("2nd request allowed", rl.allow("ip", now=100.1))
    check("3rd request blocked (over cap)", not rl.allow("ip", now=100.2))
    check("allowed again after window", rl.allow("ip", now=161.0))
    check("limit is per-key", rl.allow("other", now=100.2))
    check("disabled when per_minute=0", RateLimiter(0).allow("x"))


def test_index_bootstrap() -> None:
    print("\nIndex bootstrap (download + extract):")
    work = Path(tempfile.mkdtemp(prefix="nyay-deploy-"))
    try:
        # Fake index contents: a qdrant/ dir + bm25.pkl, mirroring the real layout.
        src = work / "src"
        (src / "qdrant").mkdir(parents=True)
        (src / "qdrant" / "collection").write_text("dummy", encoding="utf-8")
        (src / "bm25.pkl").write_bytes(b"bm25")
        # Zip the CONTENTS of src (qdrant/, bm25.pkl at the zip root) — what the runbook tells users to do.
        zpath = work / "nyaysetu-index.zip"
        with zipfile.ZipFile(zpath, "w") as z:
            for p in src.rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(src))

        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(work))
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]

        target = work / "index"
        orig_dir, orig_url = settings.index_dir, settings.index_url
        settings.index_dir = target
        settings.index_url = f"http://127.0.0.1:{port}/nyaysetu-index.zip"
        try:
            ok = ensure_index()
            check("ensure_index() returned True", ok)
            check("qdrant/ reconstructed", settings.qdrant_path.exists())
            check("bm25.pkl reconstructed", settings.bm25_file.exists())
            # Idempotent: with the index now present it must not re-download (works even if URL were dead).
            settings.index_url = "http://127.0.0.1:1/dead.zip"
            check("idempotent when already present", ensure_index())
            # Safe failure: empty dir + bad URL -> False, no crash.
            settings.index_dir = work / "empty"
            check("fails safely on bad URL", ensure_index() is False)
        finally:
            settings.index_dir, settings.index_url = orig_dir, orig_url
            srv.shutdown()
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_index_bootstrap_backslash() -> None:
    """Regression: a zip whose entries use Windows backslash separators (what PowerShell's
    Compress-Archive produces) must still unpack into a real directory tree on Linux —
    not flat files named 'qdrant\\meta.json' (which leaves Qdrant with no collection)."""
    print("\nIndex bootstrap (Windows-backslash zip -> proper tree):")
    work = Path(tempfile.mkdtemp(prefix="nyay-deploy-bs-"))
    try:
        zpath = work / "nyaysetu-index.zip"
        with zipfile.ZipFile(zpath, "w") as z:
            # Backslash arcnames, exactly as Compress-Archive writes them.
            z.writestr("qdrant\\meta.json", b"{}")
            z.writestr("qdrant\\collection\\nyaysetu_chunks\\storage.sqlite", b"data")
            z.writestr("qdrant\\.lock", b"")  # stray lock should be dropped
            z.writestr("bm25.pkl", b"bm25")

        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(work))
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]

        target = work / "index"
        orig_dir, orig_url = settings.index_dir, settings.index_url
        settings.index_dir = target
        settings.index_url = f"http://127.0.0.1:{port}/nyaysetu-index.zip"
        try:
            check("ensure_index() returned True", ensure_index())
            check("qdrant/ is a real dir", settings.qdrant_path.is_dir())
            check("nested collection file reconstructed",
                  (target / "qdrant" / "collection" / "nyaysetu_chunks" / "storage.sqlite").exists())
            check("bm25.pkl reconstructed", settings.bm25_file.exists())
            check("stray .lock dropped", not (target / "qdrant" / ".lock").exists())
        finally:
            settings.index_dir, settings.index_url = orig_dir, orig_url
            srv.shutdown()
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    print("Deployment smoke test (offline)\n" + "=" * 64)
    test_rate_limiter()
    test_index_bootstrap()
    test_index_bootstrap_backslash()
    print("=" * 64)
    print("ALL CHECKS PASSED" if _failures == 0 else f"{_failures} CHECK(S) FAILED")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
