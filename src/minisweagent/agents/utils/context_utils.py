"""Shared helpers for context-management agents (compacting, folding).

Kept as plain functions so both agents stay independent subclasses of
DefaultAgent without a common intermediate base class.
"""

from __future__ import annotations


def load_encoder(name: str = "cl100k_base"):
    """Return a tiktoken encoder, or None if tiktoken is unavailable.

    Unknown encoding names fall back to cl100k_base rather than erroring —
    token counts here only gate compaction triggers, so approximate is fine.
    """
    try:
        import tiktoken

        try:
            return tiktoken.get_encoding(name)
        except (KeyError, ValueError):
            return tiktoken.get_encoding("cl100k_base")
    except ImportError:
        return None


def message_text(msg: dict) -> str:
    from minisweagent.models.utils.content_string import get_content_string

    return get_content_string(msg) or ""


def count_tokens(messages: list[dict], encoder=None) -> int:
    """Estimate the token count of a message list. char/4 without an encoder."""
    text = "\n".join(message_text(m) for m in messages)
    if encoder is not None:
        return len(encoder.encode(text, disallowed_special=()))
    return max(1, len(text) // 4)


def context_text(msg: dict) -> str:
    """Text of ``msg`` as a later prompt carries it when the chat template drops past reasoning.

    A closed ``<think>`` block of an earlier assistant turn is not re-rendered
    (e.g. Qwen3.5), so only the text after ``</think>`` remains. A turn that
    never closed its block is rendered whole.
    """
    text = message_text(msg)
    if msg.get("role") == "assistant" and "</think>" in text:
        return text.rsplit("</think>", 1)[-1]
    return text


def count_context_tokens(messages: list[dict], encoder=None) -> int:
    """Like ``count_tokens`` but over ``context_text``: what the model will actually be shown."""
    text = "\n".join(context_text(m) for m in messages)
    if encoder is not None:
        return len(encoder.encode(text, disallowed_special=()))
    return max(1, len(text) // 4)
