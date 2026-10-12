"""Terminal cleanup, placeholders, and redaction for journey transcripts."""

from __future__ import annotations

import re

_ANSI = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"
    r"|\x1b[@-Z\\-_]"
)
PROMPT_RE = re.compile(r"(?:fx1-lite|fx1) ›")
KEY_PROMPT_RE = re.compile(r"API key:")
URL_PROMPT_RE = re.compile(r"endpoint URL")

# Fixture credentials. Not live secrets. Redacted from every published result.
FX1_KEY = "sk-journeyfixture9f3aAAAA"
LITE_KEY = "sk-journeyfixture9f3aBBBB"
KEYS = (FX1_KEY, LITE_KEY)


def strip_terminal(text: str) -> str:
    """Drop ANSI and carriage returns so regexes see the operator text."""
    cleaned = _ANSI.sub("", text).replace("\r", "")
    return cleaned


def substitute(text: str, *, stub_url: str, work: str) -> str:
    """Replace journey placeholders. ``{{work_re}}`` is regex-escaped."""
    return (
        text.replace("{{key}}", FX1_KEY)
        .replace("{{lite_key}}", LITE_KEY)
        .replace("{{stub_url}}", stub_url)
        .replace("{{work_re}}", re.escape(work))
        .replace("{{work}}", work)
    )


def redact(text: str) -> str:
    """Remove fixture keys before a reason or result is stored."""
    cleaned = text
    for key in KEYS:
        cleaned = cleaned.replace(key, "[REDACTED]")
    return cleaned


def snippet(text: str, limit: int = 220) -> str:
    """One redacted line for a failure reason."""
    flat = " ".join(redact(text).split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 3] + "..."
