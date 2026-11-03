"""Wire helpers for :class:`fx1.serve.client.HarnessClient`.

SSE parsers, header lookup, and the ``fx1`` extension object live here so
the client module stays inside its line budget. They raise
:class:`fx1.serve.client.HarnessTransportError` lazily: importing this
module must not cycle back through the client.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _transport_error(message: str) -> Exception:
    from fx1.serve.client import HarnessTransportError

    return HarnessTransportError(message)


def retry_after_s(headers: Mapping[str, str]) -> float | None:
    """Parse a Retry-After seconds hint; absent/malformed -> None."""
    for key, value in headers.items():
        if key.lower() == "retry-after":
            try:
                return max(0.0, float(value))
            except ValueError:
                return None
    return None


def fx1_opts(
    *,
    backend: str | None,
    byok: dict[str, str] | None,
    checkpoint_dir: str | Path | None,
    fallbacks: list[str] | None,
    receipt_hashes: list[str] | None,
    timeout_s: float | None,
) -> dict[str, Any]:
    """Assemble the ``fx1`` extension object every completion-surface
    method merges into its payload: backend link name, BYOK endpoint
    override, checkpoint dir, ordered fallbacks, evidence seals, and the
    backend deadline — only the knobs the caller actually set."""
    fx1: dict[str, Any] = {}
    if backend is not None:
        fx1["backend"] = backend
    if byok is not None:
        fx1["byok"] = byok
    if checkpoint_dir is not None:
        fx1["checkpoint_dir"] = str(checkpoint_dir)
    if fallbacks:
        fx1["fallbacks"] = list(fallbacks)
    if receipt_hashes:
        fx1["receipt_hashes"] = list(receipt_hashes)
    if timeout_s is not None:
        fx1["timeout_s"] = timeout_s
    return fx1


def openai_sse_chunks(body: bytes) -> list[dict[str, Any]]:
    """Parse the OpenAI SSE grammar — ``data: <json>`` frames up to the
    terminal ``[DONE]``. Fails closed when the stream ends unterminated."""
    chunks: list[dict[str, Any]] = []
    saw_done = False
    for line in body.decode().splitlines():
        if not line.startswith("data: "):
            continue
        frame = line[len("data: ") :].strip()
        if frame == "[DONE]":
            saw_done = True
            break
        chunks.append(json.loads(frame))
    if not saw_done:
        raise _transport_error("stream ended without [DONE]")
    return chunks


def hget(headers: Mapping[str, str], name: str) -> str | None:
    """Case-insensitive header lookup — transports differ in casing
    (urllib preserves the wire casing; test clients lowercase)."""
    low = name.lower()
    for key, value in headers.items():
        if key.lower() == low:
            return value
    return None


def responses_sse_events(body: bytes) -> list[dict[str, Any]]:
    """Collect Responses SSE frames until the terminal event.

    ``response.incomplete`` is the terminal event on a truncated turn
    (e.g. ``max_tool_calls``) — it ends the stream like ``completed``.
    """
    events: list[dict[str, Any]] = []
    saw_terminal = False
    for line in body.decode().splitlines():
        if not line.startswith("data: "):
            continue
        frame = json.loads(line[len("data: ") :])
        events.append(frame)
        if frame.get("type") in ("response.completed", "response.incomplete"):
            saw_terminal = True
            break
    if not saw_terminal:
        raise _transport_error("stream ended without response.completed/response.incomplete")
    return events


_REPLAY_TERMINAL_EVENTS = frozenset(
    {
        "response.completed",
        "response.incomplete",
        "response.failed",
        "response.cancelled",
    }
)


def responses_replay_events(body: bytes) -> list[dict[str, Any]]:
    """Collect Responses *replay* frames until the terminal event.

    Wider than :func:`responses_sse_events`: a replayed record can end
    in any terminal status — ``response.failed`` and ``response.cancelled``
    are terminals here (a create-time stream never emits them).
    """
    events: list[dict[str, Any]] = []
    saw_terminal = False
    for line in body.decode().splitlines():
        if not line.startswith("data: "):
            continue
        frame = json.loads(line[len("data: ") :])
        events.append(frame)
        if frame.get("type") in _REPLAY_TERMINAL_EVENTS:
            saw_terminal = True
            break
    if not saw_terminal:
        raise _transport_error(
            "replay stream ended without a terminal response.* event "
            "(response.completed/incomplete/failed/cancelled)"
        )
    return events
