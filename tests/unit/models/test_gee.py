"""Tests for gee — Liang-Zeger estimating equations."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.gee import bench_gee, gee


def _clustered(seed: int = 0, rho: float = 0.5, n_cl: int = 40, m: int = 6):
    rng = np.random.default_rng(seed)
    xs, ys, gs = [], [], []
    for c in range(n_cl):
        x_c = rng.normal(size=m)
        y_c = (
            0.2
            + 1.0 * x_c
            + rng.normal(scale=np.sqrt(rho))
            + rng.normal(scale=np.sqrt(1 - rho), size=m)
        )
        xs.append(x_c)
        ys.append(y_c)
        gs.append(np.full(m, c))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(gs)


def test_recovers_beta():
    x, y, g = _clustered()
    out = gee(x, y, g, corr="exchangeable")
    beta = np.asarray(out["beta"])
    assert abs(beta[1] - 1.0) < 0.25


def test_alpha_reflects_within_cluster_corr():
    x, y, g = _clustered(rho=0.6)
    out = gee(x, y, g, corr="exchangeable")
    assert float(out["alpha"]) > 0.3


def test_independence_matches_ols_direction():
    x, y, g = _clustered()
    out = gee(x, y, g, corr="independence")
    x_aug = np.column_stack([np.ones(x.size), x])
    b_ols = np.linalg.lstsq(x_aug, y, rcond=None)[0]
    beta = np.asarray(out["beta"])
    assert np.allclose(beta, b_ols, atol=1e-6)


def test_binomial_logit():
    rng = np.random.default_rng(5)
    n_cl, m = 50, 5
    xs, ys, gs = [], [], []
    for c in range(n_cl):
        x_c = rng.normal(size=m)
        eta = -0.5 + 1.5 * x_c
        p = 1 / (1 + np.exp(-eta))
        y_c = (rng.random(m) < p).astype(float)
        xs.append(x_c)
        ys.append(y_c)
        gs.append(np.full(m, c))
    out = gee(np.concatenate(xs), np.concatenate(ys), np.concatenate(gs), family="binomial")
    beta = np.asarray(out["beta"])
    assert 0.5 < beta[1] < 2.5


def test_fail_closed_single_cluster():
    rng = np.random.default_rng(7)
    x = rng.normal(size=20)
    y = rng.normal(size=20)
    with pytest.raises(ValueError):
        gee(x, y, np.zeros(20, dtype=np.int64))


def test_bench():
    out = bench_gee()
    assert out["synthetic_score"] == 1.0
