"""Tests for L-moments and regional frequency analysis."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.lmoments import (
    bench_lmoments,
    discordancy,
    gev_lmom,
    gpa_lmom,
    heterogeneity,
    lmom_fit,
    lmoments,
    normal_lmom,
    pwm,
)


def test_pwm_uniform_bounds():
    rng = np.random.default_rng(0)
    x = rng.random(500)
    b = pwm(x, 4)
    # U(0,1): b_r = E[X F^r] = 1/(r+2)
    for r in range(4):
        assert abs(b[r] - 1.0 / (r + 2)) < 0.04


def test_lmoments_normal():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(5000) * 2.0 + 1.0
    lm = lmoments(x)
    assert abs(lm["t3"]) < 0.03  # symmetric
    assert abs(lm["lam2"] * np.sqrt(np.pi) - 2.0) < 0.1


def test_gev_lmom_recovers_shape():
    rng = np.random.default_rng(2)
    y = -np.log(rng.random(3000))
    x = 1.0 + 1.5 * (1.0 - y ** (-0.2)) / (-0.2)
    fit = gev_lmom(x)
    assert abs(fit["k"] - (-0.2)) < 0.08
    assert abs(fit["sigma"] - 1.5) < 0.2


def test_gpa_and_normal_families():
    rng = np.random.default_rng(3)
    # GPA k=0 (exponential): quantile xi - sigma*log(1-F)
    x = 0.5 - 1.0 * np.log(rng.random(2000))
    fit = gpa_lmom(x)
    assert abs(fit["sigma"] - 1.0) < 0.15
    n = normal_lmom(rng.standard_normal(2000))
    assert abs(n["mu"]) < 0.1
    assert abs(n["sigma"] - 1.0) < 0.1
    with pytest.raises(ValueError):
        lmom_fit(x, "bogus")


def test_heterogeneity_homogeneous_region():
    rng = np.random.default_rng(4)
    region = [rng.standard_normal(60) * 2.0 for _ in range(6)]
    het = heterogeneity(region, n_sim=30, seed=4)
    assert np.isfinite(het["h1"])


def test_discordancy_and_fail_closed():
    rng = np.random.default_rng(5)
    region = [rng.standard_normal(50) for _ in range(6)]
    region[0] = region[0] * 5.0  # discordant site
    d = discordancy(region)
    assert d.shape == (6,)
    with pytest.raises(ValueError):
        lmoments(np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        pwm(np.array([np.nan] * 10))


def test_bench():
    res = bench_lmoments()
    assert res["synthetic_score"] == 1.0
    assert res["synthetic_gev_sigma_err"] < 0.05
