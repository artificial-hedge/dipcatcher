"""Mapping-key scanner shared by lab catalog and fx-1 audit gates.

Splits camelCase and hyphenated keys into lowercase segments, then fails
closed on forbidden research-headline tokens (sharpe / sortino / calmar / pnl /
nav). ``pnl`` and ``nav`` also match as substrings inside a segment so
``totalPnl`` and ``unrealizedpnl`` cannot evade underscore-only splitting.
"""

from __future__ import annotations

import re

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z])")
_SUBSTRING_TOKENS = frozenset({"pnl", "nav"})


def expand_mapping_key_segments(key: str) -> list[str]:
    """Lowercase segments from a JSON mapping key."""
    folded = _CAMEL_BOUNDARY.sub("_", str(key))
    folded = folded.lower().replace("-", "_")
    return [part for part in folded.split("_") if part]


def mapping_key_forbidden(key: str, forbidden: frozenset[str]) -> bool:
    """True when *key* carries a forbidden headline metric token."""
    for segment in expand_mapping_key_segments(key):
        if segment in forbidden:
            return True
        for token in _SUBSTRING_TOKENS:
            if token in forbidden and token in segment:
                return True
    return False
