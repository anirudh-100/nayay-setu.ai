"""Messaging channels — pick the delivery backend by config (default: console).

Mirrors app.services.llm_service.get_llm(): the safe/local option is the default
and a real cloud backend is opt-in via a single setting, so nothing leaves the
machine unless explicitly enabled.
"""
from __future__ import annotations

from app.channels.base import (
    InboundMessage,
    MessagingError,
    MessagingProvider,
    OutboundMessage,
)
from app.channels.console import ConsoleProvider
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# One process-wide provider instance. ConsoleProvider keeps an in-memory log of
# what it "sent", so a single shared instance is what a debug view would inspect.
_provider: MessagingProvider | None = None


def get_messaging_provider() -> MessagingProvider:
    """Return the configured channel provider (cached). 'whatsapp' selects the Meta
    Cloud API backend; anything else (default 'console') stays fully local."""
    global _provider
    if _provider is not None:
        return _provider

    choice = (settings.messaging_provider or "console").strip().lower()
    if choice == "whatsapp":
        from app.channels.whatsapp import WhatsAppCloudProvider

        _provider = WhatsAppCloudProvider()
        logger.info("Messaging provider: WhatsApp Cloud API (live)")
    else:
        _provider = ConsoleProvider()
        logger.info("Messaging provider: console (local, no messages sent)")
    return _provider


__all__ = [
    "InboundMessage",
    "OutboundMessage",
    "MessagingProvider",
    "MessagingError",
    "ConsoleProvider",
    "get_messaging_provider",
]
