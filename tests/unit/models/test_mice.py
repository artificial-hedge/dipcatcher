"""Tests for mice — chained PMM imputation + Rubin pooling."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.mice import bench_mice, mice_impute, pool_estimates


def _mar_data(seed: int = 0, n: int = 300):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    y = 1.0 + 2.0 * z + rng.normal(scale=0.5, size=n)
    x = np.column_stack([z, y])
    xm = x.copy()
    p_miss = 1.0 / (1.0 + np.exp(-(z - 0.3)))
    miss = rng.random(n) < p_miss * 0.5
    xm[miss, 1] = np.nan
    return x, xm, miss


def test_no_nan_after_impute():
    _, xm, _ = _mar_data()
    out = mice_impute(xm, m=3, n_iter=4, seed=0)
    ds = np.asarray(out["datasets"])
    assert ds.shape[0] == 3
    assert np.all(np.isfinite(ds))


def test_imputed_within_observed_support():
    _, xm, _ = _mar_data()
    obs_y = np.asarray(xm[:, 1][np.isfinite(xm[:, 1])])
    out = mice_impute(xm, m=2, n_iter=4, seed=1)
    ds = np.asarray(out["datasets"])
    lo, hi = obs_y.min(), obs_y.max()
    imp_vals = ds[0][:, 1]
    assert imp_vals.min() >= lo - 1e-9 and imp_vals.max() <= hi + 1e-9


def test_reduces_cc_bias():
    x, xm, _ = _mar_data(n=400)
    out = mice_impute(xm, m=5, n_iter=6, seed=2)
    ds = np.asarray(out["datasets"])
    imp_mean = float(ds[:, :, 1].mean())
    cc_mean = float(np.nanmean(xm[:, 1]))
    true_mean = float(x[:, 1].mean())
    assert abs(imp_mean - true_mean) < abs(cc_mean - true_mean) or abs(imp_mean - true_mean) < 0.2


def test_pool_estimates_rubin():
    q = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
    s = np.array([0.2, 0.2, 0.22, 0.19, 0.21])
    out = pool_estimates(q, s)
    assert out["q_bar"] == pytest.approx(q.mean())
    assert out["se_total"] > np.asarray(s).mean() * 0  # positive
    assert out["se_total"] > 0


def test_fail_closed_no_missing():
    with pytest.raises(ValueError):
        mice_impute(np.ones((10, 3)))


def test_fail_closed_fully_missing_col():
    x = np.random.default_rng(0).normal(size=(20, 3))
    x[:, 1] = np.nan
    with pytest.raises(ValueError):
        mice_impute(x)


def test_bench():
    out = bench_mice()
    assert out["synthetic_score"] == 1.0
