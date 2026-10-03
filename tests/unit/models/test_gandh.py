"""Tukey g-and-h distribution tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.gandh import (
    bench_gandh,
    gandh_letter_value,
    gandh_moments,
    gandh_quantile,
    gandh_simulate,
)


def test_quantile_g0_h0_is_gaussian():
    # g=0, h=0 -> N(A, B^2)
    q = gandh_quantile(np.array([0.25, 0.5, 0.75]), 1.0, 2.0, 0.0, 0.0)
    from scipy import stats

    np.testing.assert_allclose(q, 1.0 + 2.0 * stats.norm.ppf([0.25, 0.5, 0.75]))


def test_moments_gaussian_limit():
    out = gandh_moments(0.0, 1.0, 1e-9, 0.0)
    assert abs(out["mean"]) < 1e-6
    assert abs(out["var"] - 1.0) < 1e-6


def test_moments_positive_skew():
    out = gandh_moments(0.0, 1.0, 0.5, 0.05)
    assert out["skew"] > 0.0
    assert out["excess_kurt"] > 0.0


def test_letter_value_recovers_g():
    rng = np.random.default_rng(0)
    x = gandh_simulate(0.0, 1.0, 0.5, 0.06, 3000, rng)
    fit = gandh_letter_value(x)
    assert abs(fit["g"] - 0.5) < 0.15
    assert fit["h"] >= 0.0
    assert abs(fit["a"]) < 0.3


def test_letter_value_symmetric():
    rng = np.random.default_rng(1)
    x = rng.normal(size=2000)
    fit = gandh_letter_value(x)
    assert abs(fit["g"]) < 0.1


def test_moments_requires_h_lt_1():
    with pytest.raises(ValueError):
        gandh_moments(0.0, 1.0, 0.2, 1.5)


def test_input_validation():
    with pytest.raises(ValueError):
        gandh_letter_value(np.ones(20))


def test_bench_passes():
    out = bench_gandh(seed=5)
    assert out["synthetic_g_err"] < 0.2
    assert out["synthetic_h_err"] < 0.12
    assert out["synthetic_q_linf_iqr"] < 0.6
    assert out["synthetic_score"] == 1.0
