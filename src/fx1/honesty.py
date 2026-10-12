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

Documented matching boundary: a headline claim is a digit-carrying numeric
value adjacent to the token, in either order, through connective phrasing.
Separated letters (spaces, dots, hyphens) and compatibility digits are folded
before that test. Spelled-out numbers ("sharpe of two point one") and a
forbidden token glued onto a prefix ("mySharpe 2.1") stay outside the gate.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

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
        r"\bguaranteed\s+[-+$]?\d[\d,.%$]*\s+(?:returns?|alpha|profits?|gains?)\b",
        r"\b(?:returns?|profits?|gains?)\s+of\s+[-+$]?\d[\d,.%$]*\s+"
        r"(?:are|is|were)\s+guaranteed\b",
        r"\blive[-\s]+(?:account|performance)\b(?:\s+[A-Za-z]+){0,4}\s+[-+$]?\d",
        r"\bin\s+live[-\s]+trading\b(?:\s+[A-Za-z]+){0,6}\s+"
        r"(?:made|gained|returned|posted)\s+[-+$]?\d",
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


# Fullwidth and superscript/subscript digits that NFKC glues onto a letter
# ("sharpe²", "ｓｈａｒｐｅ２") must keep a boundary so the token stays a word.
_COMPAT_DIGIT = re.compile(
    "(?<=[A-Za-z\uff21-\uff3a\uff41-\uff5a])"
    "([\u00b2\u00b3\u00b9\u2070-\u2079\u2080-\u2089\uff10-\uff19])"
)
# "s.h.a.r.p.e" / "N.A.V." / "p/l" — collapse only when the letters are a
# forbidden token, so ordinary abbreviations ("U.S.") stay intact.
_LETTER_RUN = re.compile(r"\b(?:[A-Za-z][\W_]+){1,}[A-Za-z]\b")
_COLLAPSE_TO = frozenset(
    {
        "sharpe",
        "sharpes",
        "sortino",
        "sortinos",
        "calmar",
        "calmars",
        "nav",
        "navs",
        "pnl",
        "pnls",
        "pl",
        "pls",
    }
)
_COLLAPSE_ALIAS = {"pl": "pnl", "pls": "pnls"}


def _separate_compat_digits(text: str) -> str:
    return _COMPAT_DIGIT.sub(r" \1", text)


def _collapse_separated(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        letters = "".join(ch for ch in match.group(0) if ch.isalpha())
        key = letters.casefold()
        if key in _COLLAPSE_TO:
            return _COLLAPSE_ALIAS.get(key, letters)
        return match.group(0)

    return _LETTER_RUN.sub(repl, text)


def _normalize(text: str) -> str:
    """Fold a text for gate matching: NFKC compatibility fold (fullwidth,
    mathematical alphanumerics, ligatures), strip invisible formatting chars,
    map Cyrillic/Greek homoglyphs, collapse spaced and separated letters."""
    # Fold lookalikes before ASCII-only boundary and spaced-letter handling:
    # a Cyrillic final e must preserve the same boundary as a Latin e.
    folded = _separate_compat_digits(text)
    folded = unicodedata.normalize("NFKC", folded).translate(_HOMOGLYPHS)
    folded = _FUSION_BREAK.sub(" ", folded)
    folded = _SPACED_LETTERS.sub("", _FORMAT_CHARS.sub("", folded))
    return _collapse_separated(folded)


# Token spellings beyond the literal token. "pnl" admits the spelled forms
# a writer actually uses: p&l, p+l, p and l.
_TOKEN_SPELLINGS: dict[str, str] = {
    "pnl": r"(?:profit\s*(?:and|&)\s*loss|p\s*(?:n|&|\+|/|and)\s*l|p\s*l)",
}
# Optional snake/camel "ratio" and the existing plural/possessive inflection.
_INFLECTION = r"(?:_?ratio)?(?:['’]?s)?"

# Connective phrasing allowed between the token and the numeric claim —
# the vocabulary a headline uses. Words outside this set end the adjacency
# (bare discussion of the metric is not a claim).
_CONNECTOR = (
    r"(?:of|=|:|is|was|were|are|at|to|reads?|hits?|reached?|posts?|posted|"
    r"lands?|landed|clocks?|clocked|prints?|printed|records?|recorded|"
    r"logs?|logged|stands?|stood|sits?|sat|runs?|ran|"
    r"came\s+(?:in|out)(?:\s+at)?|equals?|"
    r"the|a|an|this|that|its|our|your|their|my|about|roughly|approximately|"
    r"around|over|under|above|below|current(?:ly)?|latest|reported|expected|"
    r"projected|implied|delivered|generated|produced|whole|all|entire|same|"
    r"given|first|last|single|rolling|trailing|net|gross|calendar|fiscal|"
    r"respective|corresponding|for|per|rose|fell|grew|"
    r"[\"'«»“”‘’]|[^\w\s]+)"
)

# Words that may sit between the token and the connector ("Sharpe ratio of",
# "pnl for the quarter", "the strategy's sharpe stood at").
_BRIDGE = (
    r"(?:ratio|score|value|reading|measure|level|figure|number|metric|multiple|"
    r"returns?|performance|results?|strategy|model|fund|portfolio|position|"
    r"trade|run|series|grid|bench|backtest|quarter|month|year|week|period|"
    r"window|horizon|vintage|cohort|account|sleeve|book|desk|panel)\b"
)


class Fx1HonestyError(ValueError):
    """Raised when an fx-1 output violates the lab honesty contract."""

    def __init__(self, message: str = "", *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


_REV_NUMBER = r"(?<![\w.])[-+$]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?"
_REV_GAP = r"(?:ratio|score|value|reading|of|is|was|were|are|at|the|a|an|its|our|to|equals?)"


def _forward_headline(text: str, spelling: str) -> bool:
    pattern = re.compile(
        rf"\b(?:{spelling}){_INFLECTION}\b\s*(?:(?:{_BRIDGE}|{_CONNECTOR})\s*){{0,6}}"
        rf"[-+$]?\d[\d,.%$]*",
        re.IGNORECASE,
    )
    return pattern.search(text) is not None


def _reverse_headline(text: str, spelling: str) -> bool:
    """Number then token. Sentence punctuation is not a bridge, so a later
    mention of the metric does not inherit an unrelated figure."""
    pattern = re.compile(
        rf"{_REV_NUMBER}\s*(?:(?:{_REV_GAP})\s+){{0,3}}"
        rf"\b(?:{spelling}){_INFLECTION}\b",
        re.IGNORECASE,
    )
    return pattern.search(text) is not None


def _token_headline(text: str, token: str) -> bool:
    spelling = _TOKEN_SPELLINGS.get(token, re.escape(token))
    return _forward_headline(text, spelling) or _reverse_headline(text, spelling)


def _contains_forbidden_headline(text: str) -> str | None:
    """Return the offending token if *text* headlines a forbidden metric.

    A headline claim is a forbidden token (optionally inflected — plural,
    possessive, ``_ratio``) with a numeric value on either side through
    connective phrasing (e.g. "Sharpe 2.1", "2.1 sharpe", "pnl: +$4,200",
    "p&l: $4,200"). Bare discussion of why these metrics are forbidden is
    allowed. Callers pass normalized text.
    """
    for token in FORBIDDEN_HEADLINE_TOKENS:
        if _token_headline(text, token):
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
    if _unlabeled_synthetic_claim(norm):
        raise Fx1HonestyError(
            "synthetic evidence presented (with numeric claims) without "
            "an explicit SYNTHETIC label."
        )
    return text


_PRESENTING = re.compile(
    r"\b(shows?|prove[sd]?|achiev\w+|scor\w+|result\w*|accuracy|"
    r"recover\w+|performance)\b",
    re.IGNORECASE,
)
_CHUNK_SPLIT = re.compile(r"[\n.;!?]+")


def _chunk_unlabeled(chunk: str) -> bool:
    if not _SYNTHETIC_TOKEN.search(chunk) or _SYNTHETIC_LABEL.search(chunk):
        return False
    return bool(re.search(r"\d", chunk) and _PRESENTING.search(chunk))


def _unlabeled_synthetic_claim(norm: str) -> bool:
    """True when a sentence presents synthetic evidence with a numeric claim
    and no explicit SYNTHETIC label in that same sentence. A label in a
    different sentence does not cover the claim. A synthetic mention without
    a numeric claim is commentary and passes."""
    if not _SYNTHETIC_TOKEN.search(norm):
        return False
    return any(_chunk_unlabeled(chunk) for chunk in _CHUNK_SPLIT.split(norm))


def honesty_categories(text: str) -> dict[str, bool]:
    """Per-category verdicts behind :func:`validate_fx1_output`.

    Every check is evaluated independently (no short-circuit) so callers can
    report the full violation surface; ``any(result.values())`` is exactly
    the gate's raise decision. Categories: ``forbidden_headline_metric``
    (a forbidden token immediately followed by a numeric headline),
    ``live_or_synthetic_claim`` (live-performance or synthetic-as-live
    phrasing), ``unlabeled_synthetic`` (synthetic evidence presented with
    numeric claims and no SYNTHETIC label).
    """
    norm = _normalize(text)
    return {
        "forbidden_headline_metric": _contains_forbidden_headline(norm) is not None,
        "live_or_synthetic_claim": any(
            pattern.search(norm) for pattern in _FORBIDDEN_CLAIM_PATTERNS
        ),
        "unlabeled_synthetic": _unlabeled_synthetic_claim(norm),
    }


_LABEL_TOKEN = re.compile(
    r"\b(?:sharpe|sortino|calmar|nav|pnl|p\s*(?:n|&|\+|and|/)\s*l|profit\s*(?:and|&)\s*loss)\b",
    re.IGNORECASE,
)
MODEL_TEXT_WITHHELD = "answer withheld by the fx-1 honesty gate. Ask a follow-up to continue."


def _fails(text: str) -> bool:
    try:
        validate_fx1_output(text)
    except Fx1HonestyError:
        return True
    return False


def public_label(text: str) -> str:
    """Server-supplied identifier safe to print beside counts and latencies.

    A forbidden metric word is replaced. Research prose uses
    :func:`validate_fx1_output` instead; bare discussion stays allowed there.
    """
    if _fails(text) or _LABEL_TOKEN.search(_normalize(text)):
        return "withheld"
    return text


def scrub_model_text(text: str) -> str:
    """Return *text* when it passes the gate, else a fixed withheld sentence."""
    if _fails(text):
        return MODEL_TEXT_WITHHELD
    return text


def screen_piece(prior: Sequence[str], text: str) -> str:
    """Accept *text* alone and joined to *prior* model/user turns.

    A prior turn that is already a violation does not block a clean addition.
    A prior turn that is clean on its own but becomes a headline once *text*
    is appended is withheld.
    """
    if _fails(text):
        return MODEL_TEXT_WITHHELD
    if not prior:
        return text
    if not _fails("\n".join([*prior, text])):
        return text
    if _fails("\n".join(prior)):
        return text
    return MODEL_TEXT_WITHHELD


def scrub_model_transcript(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Copy *messages* with assistant/tool text that fails the gate replaced."""
    cleaned: list[dict[str, Any]] = []
    prior: list[str] = []
    for message in messages:
        cleaned.append(_scrub_message(dict(message), prior))
    return cleaned


def _scrub_message(item: dict[str, Any], prior: list[str]) -> dict[str, Any]:
    content = item.get("content")
    role = item.get("role")
    if isinstance(content, str) and role == "assistant":
        content = screen_piece(prior, content)
        item["content"] = content
        prior.append(content)
    elif isinstance(content, str) and role == "tool":
        item["content"] = screen_piece(prior, content)
    return _scrub_calls(item, prior)


def _scrub_calls(item: dict[str, Any], prior: Sequence[str]) -> dict[str, Any]:
    calls = item.get("tool_calls")
    if not isinstance(calls, list):
        return item
    item["tool_calls"] = [_scrub_call(call, prior) for call in calls]
    return item


def _scrub_call(call: Any, prior: Sequence[str]) -> Any:
    if not isinstance(call, dict):
        return call
    copied = dict(call)
    function = copied.get("function")
    if not isinstance(function, dict):
        return copied
    fn = dict(function)
    arguments = fn.get("arguments")
    if isinstance(arguments, str) and screen_piece(prior, arguments) != arguments:
        fn["arguments"] = "{}"
    copied["function"] = fn
    return copied
