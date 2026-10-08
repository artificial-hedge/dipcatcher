"""Tests for cronbach — scale reliability."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.cronbach import (
    alpha_if_deleted,
    bench_cronbach,
    cronbach_alpha,
    kr20,
    split_half,
)


def test_single_factor_high_alpha():
    rng = np.random.default_rng(0)
    theta = rng.standard_normal(300)
    x = theta[:, None] + 0.4 * rng.standard_normal((300, 8))
    assert cronbach_alpha(x)["alpha"] > 0.8


def test_noise_low_alpha():
    rng = np.random.default_rng(1)
    assert abs(cronbach_alpha(rng.standard_normal((300, 8)))["alpha"]) < 0.2


def test_alpha_if_deleted_shape():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(300)[:, None] + 0.3 * rng.standard_normal((300, 6))
    assert alpha_if_deleted(x).shape == (6,)


def test_kr20_binary():
    rng = np.random.default_rng(3)
    x = (rng.random((200, 6)) < 0.7).astype(float)
    out = kr20(x)
    assert np.isfinite(out["alpha"])


def test_split_half_correlated():
    rng = np.random.default_rng(4)
    theta = rng.standard_normal(300)
    x = theta[:, None] + 0.3 * rng.standard_normal((300, 10))
    out = split_half(x)
    assert out["spearman_brown"] > 0.6


def test_split_half_is_odd_even_not_random():
    # docstring promises an odd/even split; the old code did a seeded random
    # permutation instead. With signal planted only in odd items, the odd
    # half carries it all and r_half collapses — a random split would mix
    # signal into both halves and keep r_half high.
    rng = np.random.default_rng(5)
    theta = rng.standard_normal(400)
    x = np.zeros((400, 10))
    x[:, 0::2] = theta[:, None] + 0.2 * rng.standard_normal((400, 5))
    x[:, 1::2] = rng.standard_normal((400, 5))
    out = split_half(x)
    assert abs(out["r_half"]) < 0.3
    assert out["spearman_brown"] < 0.5


def test_fail_closed_one_item():
    with pytest.raises(ValueError):
        cronbach_alpha(np.ones((50, 1)))


def test_bench():
    out = bench_cronbach()
    assert out["synthetic_score"] == 1.0
