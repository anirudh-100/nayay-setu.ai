"""Render an AskResponse as WhatsApp text — without dropping the trust contract.

A chat reply is the riskiest place to quietly lose the things that make this engine
trustworthy. So the WhatsApp message keeps them all, just compactly:
  - the law reference and the current-law bridge (so old IPC numbers map to BNS),
  - citations *with their verification badge* (official / curated / unverified) and a
    repealed marker, so a citizen sees how much to rely on each source,
  - the escalation to free legal aid when confidence is low, and the disclaimer.

WhatsApp supports light markup (*bold*, _italic_) and caps a text message at 4096
chars, so we format with that markup and split long replies across messages.
"""
from __future__ import annotations

from app.channels.whatsapp import WHATSAPP_MAX_CHARS
from app.config import settings
from app.rag.models import Citation
from app.schemas.ask import AskResponse

_MAX_SOURCES = 3  # mobile chat: a few high-signal sources beat a long list

_VERIFICATION_BADGE = {
    "official": "✅ official",
    "curated": "📝 curated",
    "unverified": "⚠️ unverified",
}

# Generic law references that name no real section — don't render a "Law:" line for them.
_GENERIC_REFS = {"general legal guidance", "general indian law", "general", ""}


def _source_line(c: Citation) -> str:
    badge = _VERIFICATION_BADGE.get(c.verification, c.verification)
    status = " (repealed)" if c.code_status == "repealed" else ""
    return f"• {c.label}{status} — {badge}"


def _split_to_messages(text: str, limit: int = WHATSAPP_MAX_CHARS) -> list[str]:
    """Split a long body into <=limit chunks, preferring paragraph then line breaks."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = window.rfind("\n\n")
        if cut < limit // 2:
            cut = window.rfind("\n")
        if cut < limit // 2:
            cut = limit
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def format_for_whatsapp(resp: AskResponse, app_name: str | None = None) -> list[str]:
    """Turn an AskResponse into one or more WhatsApp-ready message bodies."""
    name = app_name or settings.app_name
    parts: list[str] = [f"⚖️ *{name}*", "", resp.answer.strip()]

    # Abstained answers deliberately carry no sources — don't manufacture trust.
    if not resp.abstained:
        ref = resp.law_reference.strip()
        if ref.lower() not in _GENERIC_REFS:
            parts += ["", f"📖 *Law:* {ref}"]
        if resp.current_law_note:
            parts.append(f"🔄 {resp.current_law_note.strip()}")

        sources = resp.citations[:_MAX_SOURCES]
        if sources:
            parts += ["", "🔎 *Sources:*"]
            parts += [_source_line(c) for c in sources]

        if not resp.citation_verified:
            parts += ["", "⚠️ _I couldn't confirm the cited section against my sources — please verify it._"]
        elif resp.confidence == "low":
            parts += ["", "⚠️ _Low confidence — please confirm before relying on this._"]

    action = resp.action.strip()
    if action:
        parts += ["", f"👉 *What to do:* {action}"]

    if resp.escalation:
        parts += ["", f"📞 {resp.escalation.strip()}"]

    parts += ["", f"_{resp.disclaimer.strip()}_"]

    return _split_to_messages("\n".join(parts))
