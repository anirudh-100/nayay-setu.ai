"""WhatsApp Cloud API channel — the opt-in real backend (Meta Graph API).

Off by default; enabled only when MESSAGING_PROVIDER=whatsapp and the Cloud API
credentials are configured. Until then the engine answers WhatsApp-shaped traffic
through ConsoleProvider with nothing leaving the machine.

Three concerns live here, all independently testable offline:
  - verify_signature(): authenticate inbound webhooks (Meta signs the raw body with
    your app secret, HMAC-SHA256) so a stranger can't POST fake messages.
  - parse_inbound(): turn a Cloud API webhook payload into InboundMessage objects,
    ignoring the non-message events (delivery/read receipts) Meta also sends.
  - WhatsAppCloudProvider.send(): deliver a text reply via the Graph API.

References: Meta WhatsApp Cloud API (graph.facebook.com /<phone_number_id>/messages).
"""
from __future__ import annotations

import hashlib
import hmac

from app.channels.base import InboundMessage, MessagingError, MessagingProvider, OutboundMessage
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# WhatsApp text messages cap at 4096 chars; the formatter splits to fit, but we
# guard here too so a too-long body never gets silently truncated by Meta.
WHATSAPP_MAX_CHARS = 4096


def verify_signature(app_secret: str, raw_body: bytes, signature_header: str | None) -> bool:
    """True if ``signature_header`` is a valid Meta signature for ``raw_body``.

    Meta sends ``X-Hub-Signature-256: sha256=<hex>`` = HMAC-SHA256(app_secret, body).
    If no app secret is configured we cannot verify, so we fail closed (return False)
    rather than trust an unauthenticated payload — the route decides how to handle that.
    """
    if not app_secret or not signature_header:
        return False
    expected = "sha256=" + hmac.new(
        app_secret.encode("utf-8"), raw_body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header.strip())


def parse_inbound(payload: dict) -> list[InboundMessage]:
    """Extract inbound *text* messages from a WhatsApp Cloud API webhook payload.

    The webhook also fires for delivery/read statuses and non-text messages; those
    carry no answerable question, so we skip them and return only text messages.
    """
    out: list[InboundMessage] = []
    for entry in payload.get("entry", []) or []:
        for change in entry.get("changes", []) or []:
            value = change.get("value", {}) or {}
            for msg in value.get("messages", []) or []:
                if msg.get("type") != "text":
                    continue
                body = (msg.get("text", {}) or {}).get("body", "").strip()
                sender = msg.get("from", "")
                if not body or not sender:
                    continue
                out.append(
                    InboundMessage(
                        sender=sender,
                        text=body,
                        channel="whatsapp",
                        message_id=msg.get("id", ""),
                        raw=msg,
                    )
                )
    return out


class WhatsAppCloudProvider(MessagingProvider):
    """Sends replies through the Meta WhatsApp Cloud API. Constructed only when
    selected, so the default/local path never touches the network or these creds."""

    name = "whatsapp"

    def __init__(
        self,
        *,
        access_token: str | None = None,
        phone_number_id: str | None = None,
        graph_url: str | None = None,
        timeout_s: int | None = None,
    ) -> None:
        self._token = (access_token if access_token is not None else settings.whatsapp_access_token).strip()
        self._phone_id = (phone_number_id if phone_number_id is not None else settings.whatsapp_phone_number_id).strip()
        self._graph_url = (graph_url or settings.whatsapp_graph_url).rstrip("/")
        self._timeout_s = timeout_s or settings.whatsapp_timeout_s

    def send(self, message: OutboundMessage) -> None:
        if not self._token or not self._phone_id:
            raise MessagingError(
                "WhatsApp Cloud API not configured "
                "(set WHATSAPP_ACCESS_TOKEN and WHATSAPP_PHONE_NUMBER_ID)."
            )
        body = message.text
        if len(body) > WHATSAPP_MAX_CHARS:
            body = body[: WHATSAPP_MAX_CHARS - 1].rstrip() + "…"

        import httpx  # local import: only needed on the opt-in cloud path

        url = f"{self._graph_url}/{self._phone_id}/messages"
        try:
            resp = httpx.post(
                url,
                headers={"Authorization": f"Bearer {self._token}"},
                json={
                    "messaging_product": "whatsapp",
                    "recipient_type": "individual",
                    "to": message.recipient,
                    "type": "text",
                    "text": {"preview_url": False, "body": body},
                },
                timeout=self._timeout_s,
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise MessagingError(f"WhatsApp send failed: {e}") from e
        logger.info("Sent WhatsApp reply to %s (%d chars)", message.recipient, len(body))
