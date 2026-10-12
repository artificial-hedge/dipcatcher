"""Fit the outbound conversation to a character budget without splitting a turn.

The saved transcript is never edited. Only the copy sent to the model is
shortened, oldest whole turns first, and the system prompt says earlier turns
are absent so the model does not pretend to remember them. A turn is a user
message plus every assistant and tool message that follows it, so a tool call
is never separated from its result. The newest turn is always kept.

No context size is assumed for either model. The default ceiling only stops a
long session from growing without bound. When a server reports that a prompt
is too long, the ceiling for that model drops to half of what was just sent.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

ENV_MAX_CHARS = "FX1_CONTEXT_MAX_CHARS"
DEFAULT_MAX_CHARS = 400_000
MIN_MAX_CHARS = 4_000


@dataclass(frozen=True)
class FittedDialogue:
    """The dialogue to send, how many leading messages were left out, and its size."""

    messages: list[dict[str, Any]]
    omitted: int
    chars: int
    turns: int


def message_chars(message: Mapping[str, Any]) -> int:
    """Serialized size of one message, tool calls included."""
    try:
        return len(json.dumps(message, ensure_ascii=False, default=str, separators=(",", ":")))
    except (TypeError, ValueError):
        return len(str(message))


def fit_dialogue(dialogue: Sequence[Mapping[str, Any]], max_chars: int) -> FittedDialogue:
    """Drop the oldest whole turns until the rest fits ``max_chars``.

    The newest turn is kept even when it alone is larger than the ceiling.
    """
    sizes = [message_chars(message) for message in dialogue]
    starts = _turn_starts(dialogue)
    total = sum(sizes)
    first = 0
    while total > max_chars and first < len(starts) - 1:
        total -= sum(sizes[starts[first] : starts[first + 1]])
        first += 1
    begin = starts[first]
    kept = [dict(message) for message in dialogue[begin:]]
    return FittedDialogue(messages=kept, omitted=begin, chars=total, turns=len(starts) - first)


def _turn_starts(dialogue: Sequence[Mapping[str, Any]]) -> list[int]:
    starts = [0]
    for index, message in enumerate(dialogue):
        if index > 0 and message.get("role") == "user":
            starts.append(index)
    return starts


def omission_note(omitted: int) -> str:
    """System-prompt note telling the model the oldest messages are not in this request."""
    return (
        f"The earliest {omitted} messages of this conversation were left out of this "
        "request to fit the context budget; do not assume you can recall them."
    )


class ContextBudget:
    """Per-model outbound ceiling in characters."""

    def __init__(self, ceiling: int = DEFAULT_MAX_CHARS) -> None:
        self._ceiling = max(MIN_MAX_CHARS, int(ceiling))
        self._lowered: dict[str, int] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ContextBudget:
        """``FX1_CONTEXT_MAX_CHARS`` when it is a whole number, else the default."""
        source = os.environ if environ is None else environ
        raw = str(source.get(ENV_MAX_CHARS, "")).strip()
        if not (raw.isascii() and raw.isdecimal()):
            return cls()
        return cls(int(raw))

    def limit(self, model: str) -> int:
        """Characters of dialogue this model may be sent."""
        with self._lock:
            return self._lowered.get(model, self._ceiling)

    def shrink(self, model: str, sent_chars: int) -> bool:
        """Lower the ceiling below a request the server rejected. False at the floor."""
        with self._lock:
            current = self._lowered.get(model, self._ceiling)
            target = max(MIN_MAX_CHARS, min(current, max(0, sent_chars)) // 2)
            if target >= current:
                return False
            self._lowered[model] = target
            return True
