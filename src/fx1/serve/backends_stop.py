"""Stop-sequence truncation and OpenAI URL helpers for inference backends."""

from __future__ import annotations

import os


def _stop_cut(text: str, stops: tuple[str, ...] | list[str] | None) -> int:
    """Index of the earliest stop-sequence occurrence, or len(text)."""
    cut = len(text)
    for s in stops or ():
        i = text.find(s)
        if i >= 0:
            cut = min(cut, i)
    return cut


def truncate_at_stops(text: str, stops: tuple[str, ...] | list[str] | None) -> str:
    """Cut ``text`` before the earliest stop sequence — OpenAI ``stop``
    semantics, applied harness-side so every backend honors the contract
    even when the upstream doesn't (the matched sequence itself is
    excluded, per spec)."""
    return text[: _stop_cut(text, stops)]


def truncate_chunks(chunks: list[str], stops: tuple[str, ...] | list[str] | None) -> list[str]:
    """``truncate_at_stops`` over a delta stream — preserves the
    provider's chunk boundaries, emitting each chunk up to the cut and a
    slice of the straddling chunk (a stop that lands mid-delta truncates
    that delta rather than dropping whole chunks early)."""
    if not stops:
        return chunks
    cut = _stop_cut("".join(chunks), stops)
    out: list[str] = []
    pos = 0
    for chunk in chunks:
        if pos >= cut:
            break
        piece = chunk[: cut - pos]
        if piece:
            out.append(piece)
        pos += len(chunk)
    return out


def chat_completions_url(base_url: str) -> str:
    """Normalize a BYOK base to the chat-completions route."""
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def openai_sibling_url(chat_url: str, route: str) -> str:
    """Derive a sibling route from a chat-completions URL."""
    base = chat_url.rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    if base.endswith(f"/{route}"):
        return base
    return f"{base}/{route}"


def env_float(name: str, override: float | None, default: float) -> float:
    """Resolve a seconds-valued knob: kwarg beats env beats default."""
    raw = override if override is not None else os.environ.get(name)
    if raw is None or raw == "":
        return default
    val = float(raw)
    if val <= 0:
        raise ValueError(f"{name} must be positive, got {val!r}")
    return val
