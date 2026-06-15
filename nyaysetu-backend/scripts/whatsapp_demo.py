"""End-to-end WhatsApp demo through the REAL engine — console provider, no network.

Runs a few real questions through the exact pipeline a WhatsApp user would hit
(RAGService.answer → format_for_whatsapp → provider) and prints the messages that
would be delivered. Proves the channel preserves the trust contract on real law,
including a non-legal question that should abstain. Nothing is sent anywhere.

Run with the API server STOPPED (embedded Qdrant is single-process):
    python scripts/whatsapp_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.channels import ConsoleProvider, InboundMessage  # noqa: E402
from app.services.whatsapp_service import WhatsAppService  # noqa: E402

QUESTIONS = [
    "What is the punishment for murder?",
    "How do I file an FIR?",
    "Is a confession to the police admissible as evidence?",
    "What is the price of gold today?",  # should abstain — no manufactured sources
]


def main() -> int:
    provider = ConsoleProvider()
    svc = WhatsAppService(provider=provider)  # real RAGService inside

    for q in QUESTIONS:
        provider.sent.clear()
        print("\n" + "=" * 70)
        print(f"USER: {q}")
        print("-" * 70)
        svc.handle_inbound([InboundMessage(sender="demo-user", text=q)])
        for i, m in enumerate(provider.sent, 1):
            tag = f" (part {i}/{len(provider.sent)})" if len(provider.sent) > 1 else ""
            print(f"NYAYSETU{tag}:\n{m.text}")
    print("\n" + "=" * 70)
    print("Demo complete — nothing was sent over any network (console provider).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
