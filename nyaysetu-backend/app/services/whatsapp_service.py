"""WhatsApp orchestration: inbound question → grounded answer → formatted reply.

Thin glue that reuses the existing trust pipeline rather than forking it: every
WhatsApp question goes through the very same RAGService.answer() the web /ask uses,
so citations, the hallucination gate, current-law bridging, abstention and
escalation all apply identically. The channel only changes how the answer is
delivered, never how it is produced.
"""
from __future__ import annotations

from app.channels import InboundMessage, MessagingError, OutboundMessage, get_messaging_provider
from app.channels.base import MessagingProvider
from app.channels.formatting import format_for_whatsapp
from app.services.llm_service import LLMError
from app.services.rag_service import RAGService
from app.utils.logger import get_logger

logger = get_logger(__name__)

_FALLBACK_REPLY = (
    "⚖️ Sorry — I couldn't process that just now. Please try again in a moment.\n\n"
    "For urgent help, call the NALSA legal-aid helpline at 15100."
)


class WhatsAppService:
    def __init__(
        self,
        rag: RAGService | None = None,
        provider: MessagingProvider | None = None,
    ) -> None:
        self._rag = rag or RAGService()
        self._provider = provider or get_messaging_provider()

    def handle_inbound(self, messages: list[InboundMessage]) -> int:
        """Answer and reply to each inbound message. Returns the count handled.

        Each message is isolated: an engine error on one sends that user a friendly
        fallback and the rest are still processed."""
        handled = 0
        for msg in messages:
            try:
                self._reply_to(msg)
                handled += 1
            except MessagingError as e:
                logger.error("Failed delivering WhatsApp reply to %s: %s", msg.sender, e)
            except Exception:
                logger.exception("Error answering WhatsApp message from %s", msg.sender)
                self._safe_send(msg.sender, _FALLBACK_REPLY)
        return handled

    def _reply_to(self, msg: InboundMessage) -> None:
        logger.info("WhatsApp question from %s: %r", msg.sender, msg.text[:120])
        try:
            answer = self._rag.answer(msg.text)
        except LLMError as e:
            logger.error("LLM failure on WhatsApp message: %s", e)
            self._safe_send(msg.sender, _FALLBACK_REPLY)
            return

        for body in format_for_whatsapp(answer):
            self._provider.send(OutboundMessage(recipient=msg.sender, text=body, channel=msg.channel))

    def _safe_send(self, recipient: str, text: str) -> None:
        try:
            self._provider.send(OutboundMessage(recipient=recipient, text=text))
        except Exception:
            logger.exception("Failed to send fallback message to %s", recipient)
