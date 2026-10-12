"""Detect a plain-text request that should use the /superpower planner.

The match is a research verb near a research object. Slash commands,
including unicode-folded and quoted ones, and ordinary questions stay on
the chat path.
"""

from __future__ import annotations

import re
import unicodedata

from fx1.honesty import _HOMOGLYPHS

_FORMAT = re.compile("[\u00ad\u061c\u180e\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")
_QUOTED = re.compile(r"`[^`\n]*`|\"[^\"\n]*\"|'[^'\n]*'|“[^”\n]*”|‘[^’\n]*’")
_SLASH_IN_QUOTE = re.compile(r"/[A-Za-z]")
_SLASH_CMD = re.compile(r"(?m)(?:^|\s)/[A-Za-z][^\n]*")
_VERB_RE = re.compile(r"\b(?:check|score|evaluat\w*|benchmark|calibrat\w*|run)\b")
_OBJECT_RE = re.compile(
    r"\b(?:volatil\w*|qlike|crps|pinball|conformal|receipts?|"
    r"benchmarks?|famil(?:y|ies)|forecasts?)\b"
)
_GAP = 160
_OBJECT_TAIL = 32


def looks_like_orchestration(text: str) -> bool:
    """True when plain text should take the same path as ``/superpower``."""
    folded = _fold(text)
    if not folded or folded.startswith("/"):
        return False
    return _matches(_mask_commands(folded))


def _fold(text: str) -> str:
    if not isinstance(text, str):
        return ""
    folded = unicodedata.normalize("NFKC", text).translate(_HOMOGLYPHS)
    return _FORMAT.sub("", folded).strip()


def _mask_commands(text: str) -> str:
    masked = _QUOTED.sub(_drop_quoted_command, text)
    return _SLASH_CMD.sub(_keep_prefix, masked)


def _drop_quoted_command(match: re.Match[str]) -> str:
    if _SLASH_IN_QUOTE.search(match.group(0)):
        return " "
    return match.group(0)


def _keep_prefix(match: re.Match[str]) -> str:
    raw = match.group(0)
    if raw[:1].isspace():
        return raw[:1]
    return ""


def _matches(text: str) -> bool:
    folded = text.casefold()
    return any(_has_object(folded, match.end()) for match in _VERB_RE.finditer(folded))


def _has_object(text: str, start: int) -> bool:
    found = _OBJECT_RE.search(text, start, start + _GAP + _OBJECT_TAIL)
    if found is None:
        return False
    return found.start() - start <= _GAP
