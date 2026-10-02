"""Tests for hegy — seasonal unit roots + stability."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.hegy import bench_hegy, canova_hansen, hegy_test


def test_stable_seasonal_rejects():
    rng = np.random.default_rng(0)
    n, s = 160, 4
    t = np.arange(n)
    y = np.sin(2 * np.pi * t / s) + 0.2 * rng.normal(size=n)
    out = hegy_test(y, mc=200, seed=0)
    assert out["p_pi34"] < 0.10
    assert out["p_pi1"] < 0.10


def test_seasonal_rw_not_rejected():
    rng = np.random.default_rng(1)
    n, s = 160, 4
    e = rng.normal(size=n)
    y = np.empty(n)
    y[:s] = e[:s]
    for k in range(s, n):
        y[k] = y[k - s] + e[k]
    out = hegy_test(y, mc=200, seed=1)
    assert out["p_pi34"] > 0.05


def test_stats_shapes():
    rng = np.random.default_rng(2)
    y = rng.normal(size=200)
    out = hegy_test(y, mc=100, seed=3)
    for k in ("t_pi1", "t_pi2", "f_pi34", "p_pi1", "p_pi2", "p_pi34"):
        assert np.isfinite(out[k])


def test_canova_hansen_stable():
    rng = np.random.default_rng(4)
    n = 200
    t = np.arange(n)
    y = np.where(t % 4 == 0, 1.0, 0.0) + 0.1 * rng.normal(size=n)
    out = canova_hansen(y, mc=200, seed=4)
    assert out["p_value"] > 0.05


def test_canova_hansen_unstable():
    rng = np.random.default_rng(5)
    n, s = 200, 4
    walks = np.cumsum(rng.normal(size=(n, s)), axis=0).ravel("F")[:n]
    out = canova_hansen(walks, mc=200, seed=5)
    assert out["p_value"] < 0.10


def test_fail_closed_short():
    with pytest.raises(ValueError):
        hegy_test(np.ones(20))


def test_fail_closed_constant():
    with pytest.raises(ValueError):
        canova_hansen(np.ones(100))


def test_bench():
    out = bench_hegy()
    assert out["score"] == 1.0
