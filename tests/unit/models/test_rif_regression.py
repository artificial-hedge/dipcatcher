"""Tests for RIF regressions (models/rif_regression.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.rif_regression import (
    bench_rif_regression,
    rif_gini,
    rif_quantile,
    rif_regression,
    rif_variance,
    synth_rif,
    unconditional_quantile_effect,
)


def test_rif_quantile_mean_is_statistic():
    y = np.random.default_rng(3).normal(0, 1, 400)
    rif = rif_quantile(y, 0.5)
    assert abs(float(np.mean(rif)) - float(np.quantile(y, 0.5))) < 0.05


def test_rif_variance_mean_is_var():
    y = np.random.default_rng(4).normal(0, 2, 400)
    rif = rif_variance(y)
    assert abs(float(np.mean(rif)) - float(np.var(y))) < 0.3


def test_rif_gini_reasonable():
    y = np.random.default_rng(5).gamma(2.0, 1.0, 500)
    rif = rif_gini(y)
    g = float(np.mean(rif))
    assert 0.2 < g < 0.7  # gamma(2) has Gini ~0.5


def test_regression_recovers_mean_shift():
    rng = np.random.default_rng(6)
    x = rng.normal(0, 1, 500)
    y = 1.5 * x + rng.normal(0, 1, 500)
    fit = rif_regression(y, x[:, None], statistic="quantile", tau=0.5)
    beta = np.asarray(fit["beta"])
    assert 1.0 < beta[1] < 2.0
    se = np.asarray(fit["se"])
    assert beta[1] / max(se[1], 1e-12) > 3.0


def test_quantile_gradient_in_heteroskedastic():
    d = synth_rif(seed=7)
    fit90 = rif_regression(np.asarray(d["y"]), np.asarray(d["x"]), statistic="quantile", tau=0.9)
    fit50 = rif_regression(np.asarray(d["y"]), np.asarray(d["x"]), statistic="quantile", tau=0.5)
    assert float(np.asarray(fit90["beta"])[1]) > float(np.asarray(fit50["beta"])[1])


def test_unconditional_group_effect():
    rng = np.random.default_rng(8)
    g = rng.random(800) < 0.5
    y = rng.normal(0, 1, 800) + 1.0 * g
    out = unconditional_quantile_effect(y, g.astype(float), 0.75)
    assert 0.6 < out["effect"] < 1.4


def test_validation():
    with pytest.raises(ValueError):
        rif_quantile(np.ones(5), 0.5)
    with pytest.raises(ValueError):
        rif_quantile(np.random.default_rng(0).normal(size=50), 1.5)
    with pytest.raises(ValueError):
        rif_gini(np.array([-1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
    with pytest.raises(ValueError):
        rif_regression(np.ones(50), np.ones(20), statistic="bogus")
    with pytest.raises(ValueError):
        unconditional_quantile_effect(np.ones(10), np.ones(10))


def test_determinism():
    d = synth_rif(seed=9)
    a = rif_quantile(np.asarray(d["y"]), 0.8)
    b = rif_quantile(np.asarray(d["y"]), 0.8)
    assert np.array_equal(a, b)


def test_bench_keys():
    out = bench_rif_regression()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects_gradient"] == 1.0
    assert out["synthetic_determinism"] == 1.0
