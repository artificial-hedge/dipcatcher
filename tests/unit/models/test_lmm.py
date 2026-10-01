"""Tests for lmm — Laird-Ware EM mixed models."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.lmm import bench_lmm, lmm_intercept_slope, lmm_random_intercept


def _ri_data(seed: int = 0, tau2: float = 1.0, s2: float = 0.5, n_cl: int = 30, m: int = 8):
    rng = np.random.default_rng(seed)
    u = rng.normal(scale=np.sqrt(tau2), size=n_cl)
    xs, ys, gs = [], [], []
    for c in range(n_cl):
        x_c = rng.normal(size=m)
        y_c = 1.0 + 2.0 * x_c + u[c] + rng.normal(scale=np.sqrt(s2), size=m)
        xs.append(x_c)
        ys.append(y_c)
        gs.append(np.full(m, c))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(gs), u


def test_recovers_beta_and_tau2():
    x, y, g, _ = _ri_data()
    out = lmm_random_intercept(x, y, g)
    beta = np.asarray(out["beta"])
    assert abs(beta[1] - 2.0) < 0.2
    assert 0.4 < out["tau2"] < 2.5


def test_blups_track_true_effects():
    x, y, g, u = _ri_data(n_cl=40)
    out = lmm_random_intercept(x, y, g)
    corr = np.corrcoef(np.asarray(out["blups"]), u)[0, 1]
    assert corr > 0.5


def test_icc_bounds():
    x, y, g, _ = _ri_data()
    out = lmm_random_intercept(x, y, g)
    assert 0.0 <= out["icc"] <= 1.0


def test_intercept_slope_runs():
    rng = np.random.default_rng(11)
    n_cl, m = 25, 10
    xs, ys, zs, gs = [], [], [], []
    for c in range(n_cl):
        x_c = rng.normal(size=m)
        z_c = rng.normal(size=m)
        u0, u1 = rng.normal(scale=0.8), rng.normal(scale=0.4)
        y_c = 0.5 + x_c + u0 + u1 * z_c + rng.normal(scale=0.5, size=m)
        xs.append(x_c)
        ys.append(y_c)
        zs.append(z_c)
        gs.append(np.full(m, c))
    out = lmm_intercept_slope(
        np.concatenate(xs), np.concatenate(ys), np.concatenate(zs), np.concatenate(gs)
    )
    assert out["tau2_int"] > 0 and out["tau2_slope"] > 0
    d_mat = np.asarray(out["d_mat"])
    assert d_mat.shape == (2, 2)
    assert np.all(np.linalg.eigvalsh(d_mat) > 0)


def test_fail_closed_one_cluster():
    rng = np.random.default_rng(3)
    x = rng.normal(size=10)
    y = rng.normal(size=10)
    with pytest.raises(ValueError):
        lmm_random_intercept(x, y, np.zeros(10, dtype=np.int64))


def test_bench():
    out = bench_lmm()
    assert out["score"] == 1.0
