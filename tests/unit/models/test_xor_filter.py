"""Adversarial probes for xor_filter — fail-closed build."""

import random

import pytest

from quant_fund.models.xor_filter import build, contains


def test_empty_keys_raise() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build([])


def test_membership_after_build() -> None:
    rng = random.Random(3)
    keys = [rng.randrange(10**9) for _ in range(120)]
    tab, m = build(keys)
    assert all(contains(tab, m, x) for x in keys)


def test_peel_failure_raises_not_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    """If peeling never completes, build must raise — not return a dead filter."""
    import quant_fund.models.xor_filter as xf

    monkeypatch.setattr(xf, "_idx", lambda x, m: (0, m // 3, 2 * m // 3))
    # every key shares the same 3 cells → count never reaches 1 → peel fails
    with pytest.raises(RuntimeError, match="peel"):
        build([1, 2, 3])
