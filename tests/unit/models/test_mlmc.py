"""Unit tests for quant_fund.models.mlmc."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.mlmc import (
    _euler_level,
    _level_sample,
    bench_mlmc,
    mlmc_estimate,
    variance_decay_rate,
)


def _gbm(x0: float = 1.0):
    def mu(x, _t):
        return 0.05 * x

    def sig(x, _t):
        return 0.2 * x

    def payoff(x):
        return np.maximum(x - 1.0, 0.0)

    return x0, mu, sig, payoff


def test_euler_level_shape_and_seed() -> None:
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    a = _euler_level(1.0, 1.0, 32, 64, *_gbm()[1:3], rng1)
    b = _euler_level(1.0, 1.0, 32, 64, *_gbm()[1:3], rng2)
    assert a.shape == (64,)
    np.testing.assert_array_equal(a, b)


def test_level0_is_plain_payoff() -> None:
    rng = np.random.default_rng(0)
    x0, mu, sig, payoff = _gbm()
    d = _level_sample(0, 128, x0, 1.0, 8, 4, mu, sig, payoff, rng)
    assert np.all(d >= 0.0)


def test_coupled_levels_have_small_variance() -> None:
    rng = np.random.default_rng(0)
    x0, mu, sig, payoff = _gbm()
    d0 = _level_sample(0, 4000, x0, 1.0, 8, 4, mu, sig, payoff, rng)
    d2 = _level_sample(2, 4000, x0, 1.0, 8, 4, mu, sig, payoff, rng)
    # coupling shrinks the difference variance well below level-0.
    assert np.var(d2) < 0.2 * np.var(d0)


def test_mlmc_estimate_keys_and_bs_close() -> None:
    x0, mu, sig, payoff = _gbm(1.0)
    out = mlmc_estimate(x0, 1.0, mu, sig, payoff, n_levels=3, n_per_level=4000, seed=7)
    assert set(out) == {"estimate", "se", "level_means", "level_vars"}
    assert out["se"] > 0.0
    # analytic GBM call: E[max(S-1,0)] for r=0.05, sigma=0.2, T=1.
    d1 = (0.05 + 0.5 * 0.04) / 0.2
    d2 = d1 - 0.2
    from scipy.stats import norm

    exact = 1.0 * norm.cdf(d1) - 1.0 * norm.cdf(d2)
    assert abs(out["estimate"] - exact) < 5 * out["se"] + 0.05


def test_variance_decay_rate_beta() -> None:
    # Fabricate variances ~ h^-1: h = 4, 16, 64, 256 -> v = 1/h.
    v = np.array([0.5, 1 / 4.0, 1 / 16.0, 1 / 64.0, 1 / 256.0])
    beta = variance_decay_rate(v, steps0=1, factor=4)
    assert beta == pytest.approx(1.0, abs=0.05)


def test_variance_decay_rate_rejects_bad() -> None:
    with pytest.raises(ValueError):
        variance_decay_rate(np.array([1.0, 0.0, 1.0]), 1, 2)


def test_bench_mlmc_score() -> None:
    out = bench_mlmc()
    assert out["synthetic_score"] == pytest.approx(1.0)
