"""fx-1 honesty inheritance.

The lab's integrity rules (``quant_fund.research.catalog``) are enforced on
research *artifacts*. This module enforces the same rules on fx-1 *outputs*:
the agent must not emit forbidden research-headline metric claims, must not
present synthetic evidence as market evidence, and must not claim live
performance. Fail-closed: any violation raises :class:`Fx1HonestyError`.

Matching runs on a *normalized* copy of the output — NFKC folding, invisible
formatting chars, spaced-letter collapse, and a small Cyrillic/Greek
homoglyph map — so typographic disguises can't smuggle a forbidden token past
the gate. The text returned to the caller is the original, unchanged.

Documented matching boundary: a "headline claim" requires a digit-carrying
numeric value adjacent to the token through connective phrasing. Spelled-out
numbers ("sharpe of two point one") and full confusable alphabets beyond the
mapped set are outside the lexical gate's scope.
"""

from __future__ import annotations

import re
import unicodedata

# Mirrors FORBIDDEN_RESEARCH_METRIC_KEYS in quant_fund.research.catalog:
# headline P&L / ratio tokens that must never appear as fx-1 research results.
FORBIDDEN_HEADLINE_TOKENS: frozenset[str] = frozenset({"sharpe", "sortino", "calmar", "pnl", "nav"})

# Phrases that constitute a live-performance or synthetic-as-live claim.
_FORBIDDEN_CLAIM_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\blive[-\s]+(?:trading[-\s]+)?(?:p(?:&|and|n)l|profits?|returns?|gains?)\b",
        r"\b(?:p(?:&|and|n)l|profits?|returns?|gains?)\s+(?:from|of)\s+live[-\s]+(?:trading|markets?|accounts?)\b",
        r"\breal[-\s]+money\s+(?:returns?|gains?|profits?)\b",
        r"\bguaranteed\s+(?:returns?|alpha|profits?)\b",
        r"\bsynthetic\s+(?:results?|data|evidence|series|benchmarks?|backtests?)\s+"
        r"(?:shows?|proves?|demonstrates?)\s+(?:live|real|market)\b",
    )
)

# Synthetic evidence must carry an explicit SYNTHETIC label nearby.
_SYNTHETIC_TOKEN = re.compile(r"\bsynthetic\b", re.IGNORECASE)
# The label must be the explicit uppercase SYNTHETIC marker (case-sensitive).
_SYNTHETIC_LABEL = re.compile(r"\bSYNTHETIC\b")

# Invisible/zero-width characters a writer can hide inside a token. Escaped
# so the source carries no raw bidi/format control characters (B613):
# Zero-width marks, bidi controls, invisible mathematical operators, BOM,
# and soft hyphen. Include directional embeddings/isolates as well as marks.
_FORMAT_CHAR_CLASS = "\u00ad\u061c\u180e\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff"
_FORMAT_CHARS = re.compile(f"[{_FORMAT_CHAR_CLASS}]")
# A run of format chars between a letter and a digit must become a space,
# not vanish — else "sharpe\u20602.1" fuses to "sharpe2.1" and loses its boundary.
_FUSION_BREAK = re.compile(
    rf"(?<=[A-Za-z])[{_FORMAT_CHAR_CLASS}]+(?=\d)|(?<=\d)[{_FORMAT_CHAR_CLASS}]+(?=[A-Za-z])"
)
# Collapse spaces between a word-initial letter and a one-letter word:
# "s h a r p e" → "sharpe", "N A V" → "NAV". Letters only — a trailing
# space before a number ("nav's 1.9") must survive or the token fuses with
# the claim. Ordinary text ("the cat sat", "I said") is unaffected.
_SPACED_LETTERS = re.compile(r"(?<=\b[A-Za-z])\s+(?=[A-Za-z]\b)")

# Cyrillic/Greek letters that NFKC does not fold to Latin but that render
# identically. Mapped to their Latin lookalike before matching; legitimate
# Cyrillic prose can still not form a forbidden token (word boundaries hold).
_HOMOGLYPHS = str.maketrans(
    {
        "а": "a",
        "е": "e",
        "і": "i",
        "о": "o",
        "р": "p",
        "ѕ": "s",
        "с": "c",
        "х": "x",
        "у": "y",
        "м": "m",
        "т": "t",
        "н": "h",
        "к": "k",
        "в": "b",
        "д": "d",
        "з": "z",
        "А": "A",
        "Е": "E",
        "І": "I",
        "О": "O",
        "Р": "P",
        "Ѕ": "S",
        "С": "C",
        "Х": "X",
        "Μ": "M",
        "Т": "T",
        "Н": "H",
        "К": "K",
        "В": "B",
        "α": "a",
        "ο": "o",
        "ρ": "p",
        "τ": "t",
        "ν": "v",
        "ι": "i",
        "κ": "k",
        "χ": "x",
        "υ": "u",
        "Α": "A",
        "Ο": "O",
        "Ρ": "P",
        "Τ": "T",
        "Ν": "N",
        "Ι": "I",
        "Κ": "K",
        "Χ": "X",
    }
)


def _normalize(text: str) -> str:
    """Fold a text for gate matching: NFKC compatibility fold (fullwidth,
    mathematical alphanumerics, ligatures), strip invisible formatting chars,
    map Cyrillic/Greek homoglyphs, collapse spaced letters."""
    # Fold lookalikes before ASCII-only boundary and spaced-letter handling:
    # a Cyrillic final e must preserve the same boundary as a Latin e.
    folded = unicodedata.normalize("NFKC", text).translate(_HOMOGLYPHS)
    folded = _FUSION_BREAK.sub(" ", folded)
    return _SPACED_LETTERS.sub("", _FORMAT_CHARS.sub("", folded))


# Token spellings beyond the literal token. "pnl" admits the spelled forms
# a writer actually uses: p&l, p+l, p and l.
_TOKEN_SPELLINGS: dict[str, str] = {
    "pnl": r"p\s*(?:n|&|\+|and)\s*l",
}


class Fx1HonestyError(ValueError):
    """Raised when an fx-1 output violates the lab honesty contract."""


def _contains_forbidden_headline(text: str) -> str | None:
    """Return the offending token if *text* headlines a forbidden metric.

    A headline claim is a forbidden token (optionally inflected — plural,
    possessive) immediately followed by a numeric value or ratio phrasing
    (e.g. "Sharpe 2.1", "pnl: +$4,200", "navs hit 1.9", "Sharpe — 2.1",
    "p&l: $4,200"). Bare discussion of why these metrics are forbidden is
    allowed. Callers pass normalized text.
    """
    bridge = (
        r"(?:ratio|score|value|reading|measure|level|figure|number|metric|multiple|"
        r"returns?|performance|results?|strategy|model|fund|portfolio|position|"
        r"trade|run|series|grid|bench|backtest|quarter|month|year|week|period|"
        r"window|horizon|vintage|cohort|account|sleeve|book|desk|panel)\b"
    )
    connector = (
        r"(?:of|=|:|is|was|were|are|at|to|for|per|the|a|an|this|that|its|our|"
        r"your|their|my|about|roughly|approximately|around|over|under|above|"
        r"below|current(?:ly)?|latest|reported|expected|projected|stood|stands|"
        r"sits?|sat|hits?|reached|reaches|posts?|posted|lands?|landed|"
        r"clocks?|clocked|prints?|printed|records?|recorded|logs?|logged|"
        r"runs?|ran|came\s+(?:in|out)\s+at|came|rose|fell|grew|implied|"
        r"delivered|generated|produced|whole|all|entire|same|given|first|last|"
        r"single|rolling|trailing|net|gross|calendar|fiscal|respective|"
        r"corresponding|reads?|[\"']|[^\w\s]+)"
    )
    for token in FORBIDDEN_HEADLINE_TOKENS:
        spelling = _TOKEN_SPELLINGS.get(token, re.escape(token))
        pattern = re.compile(
            rf"\b{spelling}(?:['’]?s)?\b\s*(?:(?:{bridge}|{connector})\s*){{0,6}}"
            rf"[-+$]?\d[\d,.%$]*",
            re.IGNORECASE,
        )
        if pattern.search(text):
            return token
    return None


def validate_fx1_output(text: str) -> str:
    """Fail-closed validation of a single fx-1 natural-language output.

    Returns *text* unchanged when clean; raises :class:`Fx1HonestyError`
    otherwise. Applied to every agent message that leaves a session.
    """
    norm = _normalize(text)
    token = _contains_forbidden_headline(norm)
    if token is not None:
        raise Fx1HonestyError(
            f"fx-1 output headlines forbidden research metric token {token!r}; "
            "research results are proper scores, not Sharpe/P&L headlines."
        )
    for pattern in _FORBIDDEN_CLAIM_PATTERNS:
        if pattern.search(norm):
            raise Fx1HonestyError(
                "fx-1 output contains a live-performance or synthetic-as-live "
                "claim; the lab's evidence gates forbid this."
            )
    if _SYNTHETIC_TOKEN.search(norm) and not _SYNTHETIC_LABEL.search(norm):
        # The label rule targets *presentation* of synthetic results, not
        # discussion. If no numeric claim accompanies the mention, the text
        # is commentary (e.g. a refusal) and passes.
        numeric = re.search(r"\d", norm)
        presenting = re.search(
            r"\b(shows?|prove[sd]?|achiev\w+|scor\w+|result\w*|accuracy|"
            r"recover\w+|performance)\b",
            norm,
            re.IGNORECASE,
        )
        if numeric and presenting:
            raise Fx1HonestyError(
                "synthetic evidence presented (with numeric claims) without "
                "an explicit SYNTHETIC label."
            )
    return text
