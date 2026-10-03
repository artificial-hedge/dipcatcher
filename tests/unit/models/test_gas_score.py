"""Tests for score-driven GAS models (models/gas_score.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.gas_score import (
    bench_gas_score,
    gas_filter,
    gas_poisson,
    gas_t_fisher,
    gas_t_mle,
    gas_t_score,
    gas_t_volatility,
    synth_gas_poisson,
    synth_gas_t,
)


def test_gas_filter_shape():
    y = np.random.default_rng(0).standard_normal(100)
    th = gas_filter(y, lambda yy, tt: yy * yy / max(tt, 1e-6) - 1.0, 0.0, 0.1, 0.9, scaling="unit")
    assert th.shape == (100,)
    assert np.all(np.isfinite(th))


def test_gas_filter_stationary():
    # constant variance input: θ should settle near a fixed point
    rng = np.random.default_rng(1)
    y = rng.standard_normal(500)
    th = gas_t_volatility(y, alpha=0.05, beta=0.95)
    assert np.all(th > 0)
    assert abs(np.log(th[-1] / th[-50])) < 0.3


def test_gas_t_volatility_tracks_regime():
    d = synth_gas_t(n=600, seed=7)
    th = gas_t_volatility(d["y"], alpha=0.1, beta=0.95)
    corr = np.corrcoef(th[50:], d["sig2_true"][50:])[0, 1]
    assert corr > 0.4


def test_gas_t_score_signs():
    # large |y| at fixed θ → positive score (increase variance)
    s_big = gas_t_score(y=5.0, sigma2=1.0, nu=8.0)
    s_small = gas_t_score(y=0.01, sigma2=1.0, nu=8.0)
    assert s_big > 0 > s_small
    assert gas_t_fisher(1.0, 8.0) > 0


def test_gas_poisson_tracks_step():
    d = synth_gas_poisson(n=400, seed=3)
    lam = np.exp(gas_poisson(d["y"], alpha=0.15, beta=0.9))
    assert np.mean(lam[300:]) > np.mean(lam[50:150])
    corr = np.corrcoef(lam, d["lam_true"])[0, 1]
    assert corr > 0.3


def test_gas_mle_improves_on_static():
    d = synth_gas_t(n=400, seed=5)
    fit = gas_t_mle(d["y"])
    v0 = float(np.var(d["y"]))
    ll_stat = np.sum(sstats_t_logpdf(d["y"], v0))
    assert -float(fit["nll"]) > ll_stat


def sstats_t_logpdf(y: np.ndarray, v: float) -> np.ndarray:
    from scipy import stats

    return stats.t.logpdf(y, df=8.0, scale=np.sqrt(v))


def test_gas_scaling_modes():
    y = np.random.default_rng(2).standard_normal(200)
    for sc in ("unit", "inv_sqrt_fisher", "inv_fisher"):
        th = gas_t_volatility(y, scaling=sc)
        assert np.all(np.isfinite(th))
    with pytest.raises(ValueError):
        gas_t_volatility(y, scaling="bogus")


def test_validation():
    with pytest.raises(ValueError):
        gas_t_volatility(np.zeros(5))
    with pytest.raises(ValueError):
        gas_t_volatility(np.zeros(50), nu=1.5)
    with pytest.raises(ValueError):
        gas_poisson(-np.ones(50))
    with pytest.raises(ValueError):
        gas_filter(np.zeros(5), lambda a, b: 0.0, 0.0, 0.1, 0.9)


def test_bench_keys():
    out = bench_gas_score()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_sig2_corr"] > 0.5
    assert out["synthetic_lr_vs_static"] > 10
    assert out["synthetic_lam_corr"] > 0.3
    assert out["synthetic_determinism"] == 1.0
