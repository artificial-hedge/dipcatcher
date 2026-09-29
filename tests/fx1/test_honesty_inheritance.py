"""Blocking honesty inheritance: fx-1's forbidden set must never drift
from the lab's catalog. If the lab adds a forbidden research metric key,
this suite fails until fx-1 inherits it."""

import pytest

from fx1.honesty import FORBIDDEN_HEADLINE_TOKENS, Fx1HonestyError, validate_fx1_output
from quant_fund.research.catalog import FORBIDDEN_RESEARCH_METRIC_KEYS


def test_honesty_forbidden_tokens_mirror_lab_catalog() -> None:
    # Full bidirectional equality — not just containment in one direction.
    assert FORBIDDEN_HEADLINE_TOKENS == FORBIDDEN_RESEARCH_METRIC_KEYS


def test_honesty_mirror_is_exact_in_both_directions() -> None:
    # Redundant-by-design pins: a one-sided containment bug (e.g. fx-1 gains a
    # token the lab lacks, or silently drops one) must fail loudly either way.
    missing_from_fx1 = FORBIDDEN_RESEARCH_METRIC_KEYS - FORBIDDEN_HEADLINE_TOKENS
    assert not missing_from_fx1, f"fx-1 is missing lab keys: {sorted(missing_from_fx1)}"
    extra_in_fx1 = FORBIDDEN_HEADLINE_TOKENS - FORBIDDEN_RESEARCH_METRIC_KEYS
    assert not extra_in_fx1, f"fx-1 blocks tokens the lab does not: {sorted(extra_in_fx1)}"


def test_honesty_token_set_shape() -> None:
    # Immutable, normalized (already-casefolded, trimmed) string keys — so the
    # equality pin cannot pass on differently-shaped sets.
    assert isinstance(FORBIDDEN_HEADLINE_TOKENS, frozenset)
    assert FORBIDDEN_HEADLINE_TOKENS, "the mirrored set must never be empty"
    for token in FORBIDDEN_HEADLINE_TOKENS:
        assert isinstance(token, str)
        assert token == token.casefold(), f"{token!r} is not casefolded"
        assert token == token.strip(), f"{token!r} carries surrounding whitespace"


@pytest.mark.parametrize("key", sorted(FORBIDDEN_RESEARCH_METRIC_KEYS))
def test_honesty_every_lab_forbidden_key_rejected_when_headlined(key: str) -> None:
    with pytest.raises(Fx1HonestyError):
        validate_fx1_output(f"The strategy achieved {key}: 2.35 over the panel.")


def test_honesty_no_unmirrored_fx1_only_tokens() -> None:
    extra = FORBIDDEN_HEADLINE_TOKENS - FORBIDDEN_RESEARCH_METRIC_KEYS
    assert not extra, f"fx-1 blocks tokens the lab does not: {sorted(extra)}"
