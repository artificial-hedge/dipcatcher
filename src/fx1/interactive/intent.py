"""Detect a plain-text request that should use the /superpower planner.

The match is a research verb near a research object. Slash commands and
ordinary questions stay on the chat path.
"""

from __future__ import annotations

import re

_ORCHESTRATION = re.compile(
    r"\b(?:check|score|evaluat\w*|benchmark|calibrat\w*|run)\b"
    r"[\s\S]{0,160}\b(?:volatil\w*|qlike|crps|pinball|conformal|receipts?|"
    r"benchmarks?|famil(?:y|ies)|forecasts?)\b",
    re.IGNORECASE,
)


def looks_like_orchestration(text: str) -> bool:
    """True when plain text should take the same path as ``/superpower``."""
    stripped = text.strip()
    if not stripped or stripped.startswith("/"):
        return False
    return _ORCHESTRATION.search(stripped) is not None
