"""Proximity-window forbidden-headline matcher (LH008 core — audit F3 fix).

The old ``fx1/honesty.py`` regex requires the number to follow the token
*immediately* (``\\btoken\\b\\s*(?:of|=|:)?\\s*[-+$]?\\d``), so "Sharpe ratio
came in at 2.1", "the model's P&L was $4,200", or "hit rate — 61 percent"
sail through. This matcher instead:

1. Unicode-normalizes (NFKC) and casefolds the text.
2. Expands each forbidden token into aliases (``p&l``, ``net asset value``...).
3. Flags any alias occurrence within ``PROXIMITY_WINDOW`` chars of a numeric
   literal (incl. ``2.1``, ``$4,200``, ``61%``, ``61 percent``).
4. Second stage: any paragraph containing a forbidden token AND >= 2 numeric
   literals flags (numeric-density check).

``FORBIDDEN_HEADLINE_TOKENS`` is sourced from
``quant_fund.research.catalog.FORBIDDEN_RESEARCH_METRIC_KEYS`` (the catalog
constant — no copy; the fx1 drift-guard test keeps ``fx1/honesty.py``
honest). The import is lazy to respect the PROOFCORE layering contract
(leakage top-level imports are limited to contracts + schemas, §1.3).
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

_TOKEN_FALLBACK: frozenset[str] = frozenset({"sharpe", "sortino", "calmar", "pnl", "nav"})


def _load_forbidden_tokens() -> frozenset[str]:
    """Import the catalog constant lazily; fall back to the pinned set."""
    try:
        from quant_fund.research.catalog import FORBIDDEN_RESEARCH_METRIC_KEYS
    except (ImportError, AttributeError):
        return _TOKEN_FALLBACK
    return frozenset(str(tok) for tok in FORBIDDEN_RESEARCH_METRIC_KEYS)


FORBIDDEN_HEADLINE_TOKENS: frozenset[str] = _load_forbidden_tokens()

_TOKEN_ALIASES: dict[str, tuple[str, ...]] = {
    "pnl": ("pnl", "p&l", "p/l", "profit and loss"),
    "nav": ("nav", "net asset value"),
    "sharpe": ("sharpe", "sharpe ratio"),
    "sortino": ("sortino",),
    "calmar": ("calmar",),
}

PROXIMITY_WINDOW = 80  # chars between token and numeric literal

# Numeric literal: optional currency sign, digits with optional ,/. grouping,
# optional % or "percent" suffix. Always contains at least one digit.
_NUMERIC_RE = re.compile(r"[-+]?[$€£]?\s*\d[\d,]*(?:\.\d+)?\s*(?:%|percent\b)?")

# Spelled-out numerals (ADVERSARIAL §1a-E8/F6: "Sharpe, which exceeded two",
# "net asset value peaked near two million dollars"). Optional scale word.
_NUMBER_WORD_RE = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|half)"
    r"(?:\s+(?:hundred|thousand|million|billion))?\b"
)


def _alias_pattern(alias: str) -> re.Pattern[str]:
    """Word-boundary-ish match that also works for non-word aliases (p&l)."""
    return re.compile(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])")


@lru_cache(maxsize=4)
def _compiled_aliases(
    tokens: tuple[str, ...],
) -> tuple[tuple[str, tuple[re.Pattern[str], ...]], ...]:
    """One compiled pattern set per token set. Callers pass a sorted key."""
    return tuple(
        (token, tuple(_alias_pattern(alias) for alias in _aliases_for(token))) for token in tokens
    )


def _alias_needles(tokens: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(alias for token in tokens for alias in _aliases_for(token))


@lru_cache(maxsize=4)
def _matcher_key(tokens: frozenset[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    ordered = tuple(sorted(tokens))
    # Touch the compiled cache so the first scan pays compile once.
    _compiled_aliases(ordered)
    return ordered, _alias_needles(ordered)


def _prepare(text: str) -> str | None:
    """Normalized text, or None when no forbidden alias can occur.

    An alias match requires the alias as a substring of the normalized
    text, so a miss here is a miss for every rule that uses these aliases.
    """
    if not text:
        return None
    if text.isascii():
        # ASCII has no format characters. casefold covers the evasion of
        # uppercase tokens without NFKC.
        norm = text.casefold()
    else:
        norm = _normalize(text)
    _ordered, needles = _matcher_key(FORBIDDEN_HEADLINE_TOKENS)
    if any(needle in norm for needle in needles):
        return norm
    return None


def _normalize(text: str) -> str:
    # NFKC, then strip Cf (format) characters: zero-width spaces/joiners are
    # an evasion channel (ADVERSARIAL §1a-F3: "sha\u200brpe").
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Cf").casefold()


def _aliases_for(token: str) -> tuple[str, ...]:
    return _TOKEN_ALIASES.get(token, (token,))


def _patterns() -> tuple[tuple[str, tuple[re.Pattern[str], ...]], ...]:
    ordered, _needles = _matcher_key(FORBIDDEN_HEADLINE_TOKENS)
    return _compiled_aliases(ordered)


def find_forbidden_token_mentions(text: str) -> list[str]:
    """Forbidden tokens MENTIONED in *text* (no numeric-proximity requirement).

    Used by the warning-severity channels (f-string templates whose number
    arrives at runtime, spelled-out claims) where a digit literal may be
    absent by construction.
    """
    norm = _prepare(text)
    if norm is None:
        return []
    return [
        token
        for token, patterns in _patterns()
        if any(pattern.search(norm) for pattern in patterns)
    ]


def find_spelled_out_headline(text: str) -> list[str]:
    """Forbidden tokens headlined with a SPELLED-OUT number (ADVERSARIAL E8/F6).

    Same proximity window as the digit matcher; the numeric evidence is an
    English number word (``two``, ``twelve``, ``half a million``...).
    """
    norm = _prepare(text)
    if norm is None:
        return []
    hits: set[str] = set()
    for token, patterns in _patterns():
        for pattern in patterns:
            for match in pattern.finditer(norm):
                lo = max(0, match.start() - PROXIMITY_WINDOW)
                hi = min(len(norm), match.end() + PROXIMITY_WINDOW)
                if _NUMBER_WORD_RE.search(norm, lo, hi):
                    hits.add(token)
                    break
            if token in hits:
                break
    return sorted(hits)


def _proximity_headline_hits(
    norm: str,
    compiled: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...],
) -> set[str]:
    """Tokens whose alias sits within ``PROXIMITY_WINDOW`` of a numeric literal."""
    hits: set[str] = set()
    for token, patterns in compiled:
        for pattern in patterns:
            matched = False
            for match in pattern.finditer(norm):
                lo = max(0, match.start() - PROXIMITY_WINDOW)
                hi = min(len(norm), match.end() + PROXIMITY_WINDOW)
                if _NUMERIC_RE.search(norm, lo, hi):
                    hits.add(token)
                    matched = True
                    break
            if matched:
                break
    return hits


def _paragraph_headline_hits(
    norm: str,
    compiled: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...],
    hits: set[str],
) -> None:
    """Paragraph density: a token plus at least two numerics, outside the window.

    A paragraph that names a forbidden token and cites >= 2 numbers is a
    headline even when each individual number sits outside the proximity window.
    """
    if len(hits) >= len(compiled):
        return
    for paragraph in re.split(r"\n\s*\n|\n", norm):
        if len(_NUMERIC_RE.findall(paragraph)) < 2:
            continue
        for token, patterns in compiled:
            if token in hits:
                continue
            if any(pattern.search(paragraph) for pattern in patterns):
                hits.add(token)


def find_forbidden_headline(text: str) -> list[str]:
    """Return the forbidden tokens headlined in *text* (empty = clean)."""
    norm = _prepare(text)
    if norm is None:
        return []
    compiled = _patterns()
    hits = _proximity_headline_hits(norm, compiled)
    _paragraph_headline_hits(norm, compiled, hits)
    return sorted(hits)
