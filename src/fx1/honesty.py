"""fx-1 honesty inheritance.

The lab's integrity rules (``quant_fund.research.catalog``) are enforced on
research *artifacts*. This module enforces the same rules on fx-1 *outputs*:
the agent must not emit forbidden research-headline metric claims, must not
present synthetic evidence as market evidence, and must not claim live
performance. Fail-closed: any violation raises :class:`Fx1HonestyError`.

Matcher hardening (matching logic only — the token SET is mirrored from
``quant_fund.research.catalog.FORBIDDEN_RESEARCH_METRIC_KEYS`` and must never
be edited here; ``tests/fx1/test_honesty_inheritance.py`` blocks drift):

* **Unicode normalization + confusable fold** — text is NFKC-normalized,
  Cf-category characters (zero-width joiners/spaces, BOM, soft hyphen) are
  stripped, the result is casefolded, and a deterministic confusable map folds
  Cyrillic/Greek homoglyphs to ASCII. ``Ѕharpe`` / ``Shаrpe`` / full-width
  ``Ｓｈａｒｐｅ`` / ``sha\u200brpe`` all reduce to ``sharpe``.
* **Word-boundary correctness** — tokens are matched whole-word, so
  ``navigate`` / ``navel`` / ``sharpen`` / ``nav_final_*`` / ``init_nav`` do
  NOT fire, while de-spelling evasions (``S h a r p e``, ``s-h-a-r-p-e``) do.
* **Claim-connector proximity** — a token headlines a metric when a numeric
  value follows it directly or through bounded ratio/claim phrasing
  (``Sharpe 2.1``, ``Sharpe ratio came in at 2.35``, ``sharpe-ratio: 2.35``,
  ``The Sharpe is 2.4``, ``(Sharpe) = 2.4``, ``P&L was $4,200``,
  ``net asset value peaked at 1.9``). Bare discussion, refusals, and compound
  parity keys still pass — the mention-vs-claim contract is preserved.

The matcher is *vendored* here rather than imported from
``quant_fund.leakage.patterns`` to respect the ADR-0002 cross-root import pin
(``fx1`` may not import arbitrary ``quant_fund`` modules).
"""

from __future__ import annotations

import re
import unicodedata

# Mirrors FORBIDDEN_RESEARCH_METRIC_KEYS in quant_fund.research.catalog:
# headline P&L / ratio tokens that must never appear as fx-1 research results.
# DO NOT add / remove / rename tokens here — that is a coordinated cross-lane
# change with the catalog owner (the inheritance test enforces set equality).
FORBIDDEN_HEADLINE_TOKENS: frozenset[str] = frozenset({"sharpe", "sortino", "calmar", "pnl", "nav"})

# Surface spellings of each mirrored token (matching logic — NOT new tokens).
# Mirrors the alias table in ``quant_fund.leakage.patterns`` so both matchers
# recognize the same surface forms of the same token set.
_TOKEN_ALIASES: dict[str, tuple[str, ...]] = {
    "pnl": ("pnl", "p&l", "p/l", "profit and loss"),
    "nav": ("nav", "net asset value"),
    "sharpe": ("sharpe", "sharpe ratio"),
    "sortino": ("sortino",),
    "calmar": ("calmar",),
}

# Deterministic confusable fold, applied after NFKC + casefold. Covers the
# Cyrillic/Greek homoglyphs NFKC does *not* fold that can appear inside the
# forbidden tokens. Full-width and Roman-numeral forms are already folded to
# ASCII by NFKC, so they need no entry here.
_CONFUSABLES: dict[str, str] = {
    "\u0430": "a",  # Cyrillic а
    "\u0435": "e",  # Cyrillic е
    "\u043e": "o",  # Cyrillic о
    "\u0440": "p",  # Cyrillic р
    "\u0441": "c",  # Cyrillic с
    "\u0443": "y",  # Cyrillic у
    "\u0445": "x",  # Cyrillic х
    "\u0456": "i",  # Cyrillic і
    "\u0458": "j",  # Cyrillic ј
    "\u0455": "s",  # Cyrillic ѕ
    "\u04bb": "h",  # Cyrillic һ
    "\u043c": "m",  # Cyrillic м
    "\u0442": "t",  # Cyrillic т
    "\u043d": "h",  # Cyrillic н
    "\u03b1": "a",  # Greek α
    "\u03b5": "e",  # Greek ε
    "\u03bf": "o",  # Greek ο
    "\u03c1": "p",  # Greek ρ
    "\u03bd": "n",  # Greek ν
    "\u03b9": "i",  # Greek ι
    "\u03c4": "t",  # Greek τ
    "\u03c5": "u",  # Greek υ
    "\u03c7": "x",  # Greek χ
    "\u0261": "g",  # Latin small script g
    "\u0131": "i",  # Latin dotless i
}
_CONFUSABLE_TABLE: dict[int, str] = {ord(src): dst for src, dst in _CONFUSABLES.items()}

# Not preceded by a word character. ``_`` counts as a word character, matching
# ``\b`` semantics and the catalog's underscore-token key rule — so ``init_nav``
# and ``nav_final_dipcatcher`` stay mentions, not claims.
_LEFT_BOUNDARY = r"(?<![a-z0-9_])"
# Not immediately continued by a word character (blocks "navel", "sharpen",
# "nav_final", "nav2") while still allowing "nav 2.1" / "nav: 2.1".
_RIGHT_BOUNDARY = r"(?![a-z0-9_])"
# Separator run between *words* of a multi-word alias (non-empty, so the
# connector loop below always advances — no zero-width backtracking blowup).
_WORD_SEP = r"[\s\-\u2013\u2014]+"
# Separator run between *characters* of a fully de-spelled alias. Bounded on
# both sides so matching stays linear-time on adversarial input.
_CHAR_SEP = r"[\s\-_]{1,3}"
# Punctuation allowed between the connector phrasing and the number. Comma,
# semicolon and period are deliberately excluded so prose such as "sharpe, 3
# others" or "the nav. 2 of 5" stays a mention, not a claim.
_VALUE_GAP = r"[\s\-\u2013\u2014=>:)\]}'\"]*"
# Ratio nouns that turn a token mention into a metric phrase.
_RATIO_WORD = r"(?:ratio|value|number|multiple|level|score|reading|print)"
# Claim connectors allowed between the token and the numeric value.
_CONNECTOR_WORD = (
    r"(?:of|is|are|was|were|be|at|by|near|reached|hit|peaked|exceeded|stood|"
    r"clocked|printed|came|in|into|approximately|about|around|roughly|some|"
    r"equals?|measured|reported|ending|final|up|down|plus|minus)"
)
# Numeric literal: optional sign + currency, grouped digits, optional fraction
# and percent suffix. Always contains at least one digit.
_NUMERIC = r"[-+]?[\$\u20ac\u00a3]?\d[\d,]*(?:\.\d+)?(?:\s*(?:%|percent\b))?"
# Cap on the connector run: keeps the pattern deterministic and linear-time,
# and keeps far-away numbers inside a refusal from reading as a claim.
_MAX_CONNECTORS = 6

# Phrases that constitute a live-performance or synthetic-as-live claim.
_FORBIDDEN_CLAIM_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\blive p(?:&|and)l\b",
        r"\blive trading profit",
        r"\breal money (returns?|gains?|profits?)\b",
        r"\bguaranteed (returns?|alpha|profits?)\b",
        r"\bsynthetic results? (show|prove|demonstrate)s? (live|real|market)\b",
    )
)

# Synthetic evidence must carry an explicit SYNTHETIC label nearby.
_SYNTHETIC_TOKEN = re.compile(r"\bsynthetic\b", re.IGNORECASE)
# The label must be the explicit uppercase SYNTHETIC marker (case-sensitive,
# checked against the ORIGINAL text — it is an ASCII presentation rule).
_SYNTHETIC_LABEL = re.compile(r"\bSYNTHETIC\b")
_NUMERIC_ANY = re.compile(r"\d")
_PRESENTING = re.compile(
    r"\b(shows?|prove[sd]?|achiev\w+|scor\w+|result\w*|accuracy|recover\w+|performance)\b",
    re.IGNORECASE,
)


def _normalize_for_match(text: str) -> str:
    """Deterministic normalization for *matching* (never for returned text).

    NFKC → strip Cf-category (zero-width / format) characters → casefold →
    confusable fold. Idempotent and ASCII-targeted, so homoglyph and
    invisible-character evasions reduce to their plain spelling.
    """
    normalized = unicodedata.normalize("NFKC", text)
    stripped = "".join(ch for ch in normalized if unicodedata.category(ch) != "Cf")
    return stripped.casefold().translate(_CONFUSABLE_TABLE)


def _word_source(word: str) -> str:
    """Match *word* contiguous OR fully de-spelled, with word boundaries.

    The de-spelled form still needs a right boundary that rejects a word
    continuing *after* separators — ``n a v e l`` must not match ``nav`` —
    while allowing a connector word to follow (``s h a r p e ratio is 2.4``
    must match). Contiguous form: no word char immediately after. De-spelled
    form: either a separator run not followed by a letter/underscore, or a
    non-word char immediately after.
    """
    contiguous = rf"{_LEFT_BOUNDARY}{re.escape(word)}{_RIGHT_BOUNDARY}"
    if len(word) < 2:  # pragma: no cover - no single-char aliases exist
        return contiguous
    despelled = _CHAR_SEP.join(re.escape(ch) for ch in word)
    despelled_right = rf"(?:[\s\-_]{{1,3}}(?![a-z_])|{_RIGHT_BOUNDARY})"
    return rf"(?:{contiguous}|{_LEFT_BOUNDARY}{despelled}{despelled_right})"


def _alias_source(alias: str) -> str:
    """Regex source for one alias spelling, joining words on separators."""
    return _WORD_SEP.join(_word_source(word) for word in alias.split(" "))


def _headline_pattern(alias: str) -> re.Pattern[str]:
    """Compiled pattern for *alias* headlining a numeric value (claim, not mention).

    Shape: alias → optional ratio noun → bounded connector run → value gap →
    numeric literal. Only the forward direction is matched here (token, then
    number); the inverted phrasing ("2.35 was the Sharpe") is covered by
    :func:`_inverse_headline_pattern`.
    """
    return re.compile(
        rf"{_alias_source(alias)}"
        rf"(?:{_WORD_SEP}{_RATIO_WORD}(?![a-z]))?"
        rf"(?:{_WORD_SEP}{_CONNECTOR_WORD}(?![a-z])){{0,{_MAX_CONNECTORS}}}"
        rf"{_VALUE_GAP}{_NUMERIC}"
    )


# Verb connectors for the inverted phrasing ("2.35 was the Sharpe"). Narrower
# than the forward set: without "of"/"at"/"in", counting phrases like "3 sharpe
# variants" or "top 5 nav strategies" stay mentions.
_INVERSE_CONNECTOR_WORD = (
    r"(?:is|are|was|were|be|equals?|reached|hit|peaked|exceeded|stood|"
    r"clocked|printed|measured|reported|implied|realized)"
)


def _inverse_headline_pattern(alias: str) -> re.Pattern[str]:
    """Compiled pattern for a numeric value *preceding* the alias as a claim.

    Shape: numeric literal → value gap → optional verb connector → optional
    article/possessive → optional ratio noun → alias. Bounded like the forward
    pattern.
    """
    return re.compile(
        rf"{_NUMERIC}{_VALUE_GAP}"
        rf"(?:{_WORD_SEP}{_INVERSE_CONNECTOR_WORD}(?![a-z]))?"
        rf"(?:{_WORD_SEP}(?:the|a|an|its|our))(?![a-z])"
        rf"(?:{_WORD_SEP}{_RATIO_WORD}(?![a-z]))?"
        rf"{_WORD_SEP}{_alias_source(alias)}"
    )


def _aliases_for(token: str) -> tuple[str, ...]:
    return _TOKEN_ALIASES.get(token, (token,))


# Precompiled (token, pattern) pairs over the mirrored token set and its alias
# spellings, both directions. Sorted for deterministic iteration / reporting.
_HEADLINE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (token, builder(alias))
    for token in sorted(FORBIDDEN_HEADLINE_TOKENS)
    for alias in _aliases_for(token)
    for builder in (_headline_pattern, _inverse_headline_pattern)
)


class Fx1HonestyError(ValueError):
    """Raised when an fx-1 output violates the lab honesty contract."""


def _contains_forbidden_headline(text: str) -> str | None:
    """Return the offending token if *text* headlines a forbidden metric.

    A headline claim is a forbidden token followed — directly or through
    bounded ratio/claim phrasing — by a numeric value (e.g. "Sharpe 2.1",
    "Sharpe ratio came in at 2.35", "pnl: +$4,200"). Matching runs on a
    Unicode-normalized, confusable-folded copy; bare discussion of why these
    metrics are forbidden is allowed.
    """
    norm = _normalize_for_match(text)
    for token, pattern in _HEADLINE_PATTERNS:
        if pattern.search(norm):
            return token
    return None


def validate_fx1_output(text: str) -> str:
    """Fail-closed validation of a single fx-1 natural-language output.

    Returns *text* unchanged when clean; raises :class:`Fx1HonestyError`
    otherwise. Applied to every agent message that leaves a session.
    """
    token = _contains_forbidden_headline(text)
    if token is not None:
        raise Fx1HonestyError(
            f"fx-1 output headlines forbidden research metric token {token!r}; "
            "research results are proper scores, not Sharpe/P&L headlines."
        )
    # Claim patterns run on the normalized copy so a homoglyph / full-width /
    # zero-width "live P&L" cannot slip past an ASCII-only matcher.
    norm = _normalize_for_match(text)
    for pattern in _FORBIDDEN_CLAIM_PATTERNS:
        if pattern.search(norm):
            raise Fx1HonestyError(
                "fx-1 output contains a live-performance or synthetic-as-live "
                "claim; the lab's evidence gates forbid this."
            )
    # The label rule targets *presentation* of synthetic results, not
    # discussion. If no numeric claim accompanies the mention, the text is
    # commentary (e.g. a refusal) and passes. The label itself must be the
    # exact uppercase ASCII marker in the original text.
    if (
        _SYNTHETIC_TOKEN.search(norm)
        and not _SYNTHETIC_LABEL.search(text)
        and _NUMERIC_ANY.search(norm)
        and _PRESENTING.search(norm)
    ):
        raise Fx1HonestyError(
            "synthetic evidence presented (with numeric claims) without "
            "an explicit SYNTHETIC label."
        )
    return text
