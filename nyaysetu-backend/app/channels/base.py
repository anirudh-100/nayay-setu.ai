"""Messaging-channel abstractions — meet citizens where they already are.

Pillar 4 lets a person ask a legal question over a chat channel (WhatsApp first)
and get back the *same* cited, current-law, trust-framed answer the web app gives.
The channel is just transport; the engine and its trust contract are unchanged.

Design mirrors the LLM provider abstraction (OllamaClient/ClaudeClient + get_llm):
a small interface with swappable backends, defaulting to the safe/local option.
``ConsoleProvider`` (the default) only logs what it *would* send — so the whole
inbound→answer→outbound path is fully exercisable locally with nothing leaving the
machine. A real provider (WhatsApp Cloud API) is opt-in via config, exactly like
high-power mode is opt-in for the LLM.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class InboundMessage:
    """One incoming chat message, normalized across channels."""

    sender: str                 # opaque channel user id (e.g. WhatsApp wa_id / phone)
    text: str                   # the user's message body
    channel: str = "whatsapp"   # which channel it arrived on
    message_id: str = ""        # provider message id (dedupe / logging)
    raw: dict = field(default_factory=dict)  # original payload (debugging only)


@dataclass(frozen=True)
class OutboundMessage:
    """One reply to send back to a chat user."""

    recipient: str              # who to send to (same id space as InboundMessage.sender)
    text: str                   # the reply body (already channel-formatted)
    channel: str = "whatsapp"


class MessagingError(RuntimeError):
    """Raised when a channel provider fails to deliver a message."""


class MessagingProvider(ABC):
    """A channel backend that can deliver an outbound message.

    Kept deliberately tiny: the engine produces an answer, the formatter turns it
    into channel text, and the provider just delivers it. Inbound parsing and
    signature verification are channel-specific and live with each provider.
    """

    name: str = "base"

    @abstractmethod
    def send(self, message: OutboundMessage) -> None:
        """Deliver one message. Raise MessagingError on failure."""
        raise NotImplementedError
