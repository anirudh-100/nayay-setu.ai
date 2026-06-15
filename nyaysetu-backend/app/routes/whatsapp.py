"""WhatsApp Cloud API webhook.

Two endpoints, per Meta's contract:
  - GET  /webhooks/whatsapp — one-time verification handshake (echo hub.challenge
    when the verify token matches).
  - POST /webhooks/whatsapp — inbound messages. We authenticate the payload against
    the app secret, hand the work to a background task, and return 200 immediately
    (Meta retries on any non-200, so the ack must not wait on the LLM).

Security posture: when an app secret IS configured we fail closed — an unsigned or
badly-signed POST is rejected. With no secret (the local/dev default) we accept and
log a warning, so the path is testable on a laptop without Meta credentials.
"""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Query, Request
from fastapi.responses import PlainTextResponse

from app.channels.whatsapp import parse_inbound, verify_signature
from app.config import settings
from app.services.whatsapp_service import WhatsAppService
from app.utils.logger import get_logger

router = APIRouter(prefix="/webhooks", tags=["whatsapp"])
logger = get_logger(__name__)


@router.get("/whatsapp")
def verify_webhook(
    mode: str = Query("", alias="hub.mode"),
    token: str = Query("", alias="hub.verify_token"),
    challenge: str = Query("", alias="hub.challenge"),
) -> PlainTextResponse:
    """Meta calls this once when you register the webhook. Echo the challenge back
    only if the token matches our configured verify token."""
    if mode == "subscribe" and token and token == settings.whatsapp_verify_token:
        logger.info("WhatsApp webhook verified.")
        return PlainTextResponse(challenge)
    logger.warning("WhatsApp webhook verification failed (mode=%r).", mode)
    return PlainTextResponse("Verification failed", status_code=403)


@router.post("/whatsapp")
async def receive_webhook(request: Request, background: BackgroundTasks) -> PlainTextResponse:
    raw = await request.body()

    if settings.whatsapp_app_secret:
        sig = request.headers.get("X-Hub-Signature-256")
        if not verify_signature(settings.whatsapp_app_secret, raw, sig):
            logger.warning("Rejected WhatsApp webhook with bad/missing signature.")
            return PlainTextResponse("Invalid signature", status_code=403)
    else:
        logger.warning("WHATSAPP_APP_SECRET unset — accepting webhook without signature check (dev only).")

    try:
        payload = await request.json()
    except Exception:
        logger.warning("WhatsApp webhook with non-JSON body — ignoring.")
        return PlainTextResponse("ok")  # ack so Meta doesn't retry a junk payload

    messages = parse_inbound(payload)
    if messages:
        # Answer + reply off the request path so we can 200 immediately.
        background.add_task(WhatsAppService().handle_inbound, messages)
        logger.info("Queued %d WhatsApp message(s) for answering.", len(messages))

    return PlainTextResponse("ok")
