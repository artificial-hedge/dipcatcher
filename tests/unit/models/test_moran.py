"""Tests for moran — spatial autocorrelation."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.moran import bench_moran, geary_c, getis_ord_g, moran_i


def _ring_w(n=40):
    w = np.zeros((n, n))
    for i in range(n):
        for k in (1, 2):
            w[i, (i + k) % n] = 1.0
            w[i, (i - k) % n] = 1.0
    return w


def test_smooth_field_positive_i():
    rng = np.random.default_rng(0)
    n = 40
    w = _ring_w(n)
    x = np.sin(np.arange(n) * 4 * np.pi / n) + 0.1 * rng.standard_normal(n)
    out = moran_i(x, w, n_perm=199, seed=0)
    assert out["i"] > 0.3
    assert out["p"] < 0.01


def test_iid_near_expected():
    rng = np.random.default_rng(1)
    w = _ring_w(40)
    out = moran_i(rng.standard_normal(40), w, n_perm=199, seed=1)
    assert out["i"] < 0.4


def test_geary_c_below_one_for_smooth():
    n = 40
    w = _ring_w(n)
    x = np.sin(np.arange(n) * 4 * np.pi / n)
    out = geary_c(x, w, n_perm=99, seed=2)
    assert out["c"] < 0.7


def test_getis_g_runs():
    rng = np.random.default_rng(3)
    w = _ring_w(30)
    out = getis_ord_g(rng.random(30), w)
    assert np.isfinite(out["g"])


def test_fail_closed_constant():
    with pytest.raises(ValueError):
        moran_i(np.ones(10), np.ones((10, 10)))


def test_bench():
    out = bench_moran()
    assert out["synthetic_score"] == 1.0
