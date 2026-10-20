"""Mapping-key scanner shared by lab catalog and fx-1 audit gates.

Splits camelCase and other separators into lowercase segments, folds
spelled-out P&L / NAV (``profitAndLoss``, ``net_asset_value``), then fails
closed on forbidden research-headline tokens. ``unrealizedpnl`` still matches
because the segment ends in ``pnl``. ``navigate`` and ``plan`` do not.
"""

from __future__ import annotations

import re

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z])")
_SEPARATOR = re.compile(r"[^A-Za-z0-9]+")
# Longer phrases first so a trailing plural is consumed with the phrase.
_SPELLED_PHRASES: tuple[tuple[str, str], ...] = (
    ("profits and losses", "pnl"),
    ("profit and losses", "pnl"),
    ("profits and loss", "pnl"),
    ("profit and loss", "pnl"),
    ("net asset values", "nav"),
    ("net asset value", "nav"),
    ("p and l", "pnl"),
    ("p n l", "pnl"),
    ("p l", "pnl"),
)


def expand_mapping_key_segments(key: str) -> list[str]:
    """Lowercase segments from a JSON mapping key."""
    folded = _CAMEL_BOUNDARY.sub("_", str(key))
    folded = _SEPARATOR.sub("_", folded).lower()
    parts = [part for part in folded.split("_") if part]
    phrase = f" {' '.join(parts)} "
    for src, dst in _SPELLED_PHRASES:
        phrase = phrase.replace(f" {src} ", f" {dst} ")
    return phrase.split()


_SUBSTRING_TOKENS = frozenset({"pnl", "nav"})


def _substring_token_in_segment(segment: str, token: str) -> bool:
    """Match ``pnl`` / ``nav`` in a segment without ``unavailable``-style false positives."""
    if segment == token:
        return True
    if segment.endswith(token):
        return True
    needle = f"_{token}"
    return needle in f"_{segment}_"


def mapping_key_forbidden(key: str, forbidden: frozenset[str]) -> bool:
    """True when *key* carries a forbidden headline metric token."""
    for segment in expand_mapping_key_segments(key):
        if segment in forbidden:
            return True
        for token in _SUBSTRING_TOKENS:
            if token in forbidden and _substring_token_in_segment(segment, token):
                return True
    return False
