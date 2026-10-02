"""Adversarial mutation tests for the fx-1 honesty gate.

Every parametrize case below was a worked bypass of the naive token match —
typographic disguises (fullwidth, mathematical alphanumeric, zero-width
insertion, Cyrillic/Greek homoglyphs, spaced letters), inflections (plural,
possessive), alternate spellings (p&l family), and claim verbs outside the
original connector set ("hit", "came in at", "printed"). All must raise
:class:`Fx1HonestyError`; the boundary cases pin the allowed-discussion rule
(a token without an adjacent numeric claim is commentary, not a headline).
"""

from __future__ import annotations

import pytest

from fx1.honesty import Fx1HonestyError, validate_fx1_output


@pytest.mark.parametrize(
    "text",
    [
        # canonical forms still caught
        "Sharpe ratio: 2.1",
        "sharpe is 1.94",
        "calmar — 1.4",
        "nav (1.9)",
        "pnl: +$4,200",
        # inflections — \b-tokens previously missed these
        "the navs printed 1.91",
        "pnls came in at 4200",
        "the sharpes were 2.1",
        "sortinos of 0.8",
        "nav's 1.9",
        "the model's sharpe reached 2.1",
        # spelled p&l forms — the token set only knew 'pnl'
        "p&l: $4,200",
        "P+L = 4,200",
        "p and l was $4,200",
        "backtest p&l at $2.1M",
        # claim verbs outside the old connector vocabulary
        "Sharpe hit 2.1",
        "sharpe reached 2.1",
        "calmar printed 1.4",
        "nav posted 1.9",
        "sortino landed at 0.8",
        "pnl clocked 4,200",
        "sharpe recorded 2.1",
        "calmar stands at 1.4",
        "nav came in at 1.9",
        "sharpe ran 2.1",
        # typographic disguise — fullwidth / math alphanumeric
        "ｓｈａｒｐｅ ratio: 2.1",
        "ＳＨＡＲＰＥ = 2.1",
        "𝐬𝐡𝐚𝐫𝐩𝐞 is 2.1",
        "𝐧𝐚𝐯 of 1.9",
        # invisible formatting chars inside or beside the token
        "sharpe\u200b: 2.1",
        "sh\u200barpe ratio of 2.1",
        "sharpe\u20602.1",
        # spaced letters
        "s h a r p e is 2.1",
        "N A V of 1.9",
        "p n l: 4200",
        # Cyrillic / Greek homoglyphs
        "ѕhаrре: 2.1",
        "nаv is 1.9",
        "shαrpe ratio 2.1",
        # mixed classes
        "ｓｈａｒｐｅ hit 2.1",
        "S H A R P E came in at 2.1",
        # claim-pattern disguises (live-performance phrasing)
        "lіve trading pnl of $4,200",
        "live\u200b-\u200btrading returns of 4%",
        "real-\u200bmoney gains of 12%",
        # synthetic presentation without the uppercase label
        "ｓynthetic results show 92% accuracy",
        "synthetic\u200b benchmark achieves 88%",
    ],
)
def test_adversarial_headline_bypasses_fail_closed(text: str) -> None:
    with pytest.raises(Fx1HonestyError):
        validate_fx1_output(text)


@pytest.mark.parametrize(
    "text",
    [
        # token with no adjacent numeric claim — commentary, allowed
        "Sharpe is a forbidden headline metric in this lab.",
        "we report pinball and CRPS, never sharpe or nav figures.",
        "the nav discussion is about why it is excluded",
        "p&l accounting conventions differ across venues",
        "results report proper scores only",
        "s h a r p e is spelled out as a boundary case with no number",
        # numbers unrelated to a forbidden token
        "the run completed in 2.1 seconds",
        "pinball loss improved to 0.31",
        # synthetic mention without numeric presentation is commentary
        "synthetic data is used for correctness tests only",
        # properly labeled synthetic presentation passes
        "SYNTHETIC results show 92% accuracy on the fixture",
        "ＳＹＮＴＨＥＴＩＣ results show 92% accuracy",  # fullwidth label still counts
    ],
)
def test_allowed_discussion_still_passes(text: str) -> None:
    assert validate_fx1_output(text) == text


def test_normalization_preserves_returned_text() -> None:
    """The gate returns the original bytes — normalization is for matching only."""
    original = "ｓynthetic data has no numeric claim"
    assert validate_fx1_output(original) == original


@pytest.mark.parametrize(
    "text",
    [
        # A run of invisible chars must preserve the token/number boundary.
        "Sharpe\u200b\u20602.1",
        "nav\u200e\ufeff1.9",
        "pnl\u00ad\u200b4200",
        # Homoglyph folding must happen before ASCII boundary/collapse rules.
        "s h α r p e: 2.1",
        "Ν Α V of 1.9",
        "sharpе\u200b2.1",  # final letter is Cyrillic
        "sharpе\u2060\ufeff2.1",
        # Bidi and invisible mathematical formatting must not split a token.
        "sha\u202erpe: 2.1",
        "na\u2066v of 1.9",
        "live\u2063 trading returns of 4%",
        "synthe\u2063tic results show 92% accuracy",
    ],
)
def test_composed_normalization_bypasses_fail_closed(text: str) -> None:
    with pytest.raises(Fx1HonestyError):
        validate_fx1_output(text)


@pytest.mark.parametrize(
    "text",
    [
        "s h α r p e is excluded from research headlines",
        "sha\u202erpe is a forbidden metric",
        "S Y Ν T H E T I C results show 92% accuracy",
        "SYNTHETIC results show 92% accuracy\u2063 on the fixture",
    ],
)
def test_composed_normalization_preserves_allowed_text(text: str) -> None:
    assert validate_fx1_output(text) == text
