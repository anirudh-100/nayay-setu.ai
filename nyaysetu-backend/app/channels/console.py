"""Console messaging provider — the safe, local default.

Sends nothing over the network: it logs the outbound message and records it in
memory. This keeps the free/local-first promise (the whole WhatsApp path works on
a laptop with no Meta account, no public URL, nothing leaving the machine) and
makes the inbound→answer→outbound flow trivially testable.
"""
from __future__ import annotations

from app.channels.base import MessagingProvider, OutboundMessage
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ConsoleProvider(MessagingProvider):
    name = "console"

    def __init__(self) -> None:
        # Last-N sent messages, so tests / a debug view can assert what went out.
        self.sent: list[OutboundMessage] = []

    def send(self, message: OutboundMessage) -> None:
        self.sent.append(message)
        logger.info(
            "[console:%s] would send to %s (%d chars):\n%s",
            message.channel,
            message.recipient,
            len(message.text),
            message.text,
        )
