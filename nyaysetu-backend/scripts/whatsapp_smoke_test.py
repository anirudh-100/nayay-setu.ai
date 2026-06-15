"""Offline smoke test for the WhatsApp channel — no Meta account, no network, no models.

Exercises the whole inbound→answer→outbound path with everything faked at the edges:
  - signature verification (HMAC-SHA256) accepts good and rejects bad/missing,
  - inbound parsing pulls text messages and ignores status/non-text events,
  - the formatter keeps the trust contract (law ref, sources + verification badges,
    repealed marker, escalation, disclaimer) and drops sources when abstaining,
  - WhatsAppService routes a question through a FAKE engine to the console provider,
  - the webhook route verifies the GET handshake and the POST signature gate.

Usage:
    python scripts/whatsapp_smoke_test.py
"""
from __future__ import annotations

import hashlib
import hmac
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.channels import ConsoleProvider, InboundMessage, OutboundMessage  # noqa: E402
from app.channels.formatting import format_for_whatsapp  # noqa: E402
from app.channels.whatsapp import parse_inbound, verify_signature  # noqa: E402
from app.config import settings  # noqa: E402
from app.rag.models import Citation  # noqa: E402
from app.schemas.ask import DISCLAIMER, LEGAL_AID_ESCALATION, AskResponse  # noqa: E402

_failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global _failures
    if not ok:
        _failures += 1
    line = f"  [{'PASS' if ok else 'FAIL'}] {name}"
    if detail:
        line += f"  - {detail}"
    print(line)


def _answer(**kw) -> AskResponse:
    base = dict(
        answer="Murder is punishable with death or imprisonment for life.",
        law_reference="BNS Section 103",
        action="If someone is in danger, call 112 and consult a lawyer.",
        confidence="high",
        reasoning="Used BNS 103 from context",
        citations=[
            Citation(label="BNS Section 103", source_type="statute", snippet="...",
                     code_status="current", verification="official"),
            Citation(label="IPC Section 302", source_type="statute", snippet="...",
                     code_status="repealed", verification="unverified"),
        ],
        abstained=False,
        escalation=None,
        current_law_note="IPC Section 302 now corresponds to BNS Section 103.",
        citation_verified=True,
        disclaimer=DISCLAIMER,
        response_time_ms=12,
    )
    base.update(kw)
    return AskResponse(**base)


def main() -> int:
    print("WhatsApp channel smoke test (offline)\n" + "=" * 64)

    # --- Signature verification ---
    print("\nSignature verification:")
    secret, body = "app-secret", b'{"hello":"world"}'
    good = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    check("accepts a valid signature", verify_signature(secret, body, good))
    check("rejects a tampered body", not verify_signature(secret, b'{"hello":"evil"}', good))
    check("rejects a missing signature", not verify_signature(secret, body, None))
    check("fails closed with no secret", not verify_signature("", body, good))

    # --- Inbound parsing ---
    print("\nInbound parsing:")
    text_payload = {
        "entry": [{"changes": [{"value": {"messages": [
            {"from": "9199999", "id": "wamid.1", "type": "text",
             "text": {"body": "What is the punishment for murder?"}}
        ]}}]}]
    }
    msgs = parse_inbound(text_payload)
    check("extracts a text message", len(msgs) == 1 and msgs[0].sender == "9199999")
    check("reads the body", msgs and msgs[0].text.startswith("What is the punishment"))

    status_payload = {"entry": [{"changes": [{"value": {"statuses": [{"status": "delivered"}]}}]}]}
    check("ignores status events", parse_inbound(status_payload) == [])
    image_payload = {"entry": [{"changes": [{"value": {"messages": [
        {"from": "9199999", "id": "wamid.2", "type": "image", "image": {"id": "x"}}]}}]}]}
    check("ignores non-text messages", parse_inbound(image_payload) == [])

    # --- Formatter (trust contract) ---
    print("\nFormatter — substantive answer:")
    out = format_for_whatsapp(_answer(), app_name="NyaySetu")
    text = "\n".join(out)
    check("single message", len(out) == 1)
    check("shows the law reference", "📖 *Law:* BNS Section 103" in text)
    check("shows the current-law bridge", "🔄" in text and "BNS Section 103" in text)
    check("lists sources", "🔎 *Sources:*" in text)
    check("official badge present", "✅ official" in text)
    check("unverified badge present", "⚠️ unverified" in text)
    check("marks repealed source", "(repealed)" in text)
    check("includes the disclaimer", DISCLAIMER in text)

    print("\nFormatter — abstained answer (no manufactured trust):")
    ab = format_for_whatsapp(
        _answer(abstained=True, law_reference="General Legal Guidance", citations=[],
                escalation=LEGAL_AID_ESCALATION, confidence="low",
                answer="I couldn't find a reliable basis to answer this confidently.",
                current_law_note=None),
        app_name="NyaySetu",
    )
    abt = "\n".join(ab)
    check("no Sources block when abstaining", "🔎 *Sources:*" not in abt)
    check("no Law line for generic ref", "📖 *Law:*" not in abt)
    check("shows escalation to legal aid", "15100" in abt)

    print("\nFormatter — unverified citation warning + long-message split:")
    uv = "\n".join(format_for_whatsapp(_answer(citation_verified=False)))
    check("warns when citation unverified", "couldn't confirm" in uv)
    long_ans = format_for_whatsapp(_answer(answer="A " * 3000))  # ~6000 chars
    check("splits a long reply into multiple messages", len(long_ans) > 1)
    check("each chunk within WhatsApp limit", all(len(c) <= 4096 for c in long_ans))

    # --- Provider factory + service (fake engine) ---
    print("\nWhatsAppService with a fake engine -> console provider:")
    from app.services.whatsapp_service import WhatsAppService

    class _FakeRAG:
        def __init__(self) -> None:
            self.asked: list[str] = []

        def answer(self, query: str, language: str = "en") -> AskResponse:
            self.asked.append(query)
            return _answer()

    provider = ConsoleProvider()
    fake = _FakeRAG()
    svc = WhatsAppService(rag=fake, provider=provider)
    handled = svc.handle_inbound([InboundMessage(sender="9199999", text="punishment for murder?")])
    check("handled the message", handled == 1)
    check("engine was asked the question", fake.asked == ["punishment for murder?"])
    check("a reply was sent via provider", len(provider.sent) == 1)
    check("reply is addressed to the sender",
          provider.sent and provider.sent[0].recipient == "9199999")
    check("reply carries trust signals", provider.sent and "📖 *Law:*" in provider.sent[0].text)

    # --- Webhook route via TestClient (route patched to a fake; no models) ---
    print("\nWebhook route (GET handshake + POST signature gate):")
    from fastapi.testclient import TestClient

    import app.routes.whatsapp as wa_route
    from app.main import app

    captured: list[list[InboundMessage]] = []

    class _FakeService:
        def handle_inbound(self, messages):
            captured.append(messages)
            return len(messages)

    wa_route.WhatsAppService = _FakeService  # avoid constructing the real (model-loading) service

    original_token = settings.whatsapp_verify_token
    original_secret = settings.whatsapp_app_secret
    settings.whatsapp_verify_token = "verify-me"
    settings.whatsapp_app_secret = "post-secret"
    try:
        client = TestClient(app)  # no 'with' => skip lifespan/model warmup

        r = client.get("/webhooks/whatsapp", params={
            "hub.mode": "subscribe", "hub.verify_token": "verify-me", "hub.challenge": "42"})
        check("GET handshake echoes challenge on correct token", r.status_code == 200 and r.text == "42")
        r = client.get("/webhooks/whatsapp", params={
            "hub.mode": "subscribe", "hub.verify_token": "wrong", "hub.challenge": "42"})
        check("GET handshake rejects wrong token", r.status_code == 403)

        raw = json.dumps(text_payload).encode()
        sig = "sha256=" + hmac.new(b"post-secret", raw, hashlib.sha256).hexdigest()
        r = client.post("/webhooks/whatsapp", content=raw,
                        headers={"X-Hub-Signature-256": sig, "Content-Type": "application/json"})
        check("POST accepts a correctly-signed payload", r.status_code == 200)
        check("POST queued the parsed message", len(captured) == 1 and len(captured[0]) == 1)

        r = client.post("/webhooks/whatsapp", content=raw,
                        headers={"X-Hub-Signature-256": "sha256=bad", "Content-Type": "application/json"})
        check("POST rejects a bad signature", r.status_code == 403)
        check("rejected payload was not queued", len(captured) == 1)
    finally:
        settings.whatsapp_verify_token = original_token
        settings.whatsapp_app_secret = original_secret

    print("=" * 64)
    print("ALL CHECKS PASSED" if _failures == 0 else f"{_failures} CHECK(S) FAILED")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
