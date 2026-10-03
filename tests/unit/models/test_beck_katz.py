"""Beck-Katz PCSE tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.beck_katz import (
    beck_katz,
    bench_pcse,
    parks_fgls,
)


def _panel(seed=0, n_u=8, n_t=30):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n_u * n_t), rng.normal(0, 1, n_u * n_t)])
    sig = 0.5 * np.ones((n_u, n_u)) + 0.5 * np.eye(n_u)
    L = np.linalg.cholesky(sig)
    e = (L @ rng.normal(0, 1, (n_u, n_t))).ravel()
    y = X @ np.array([0.3, 1.2]) + e
    return y, X, n_u, n_t


def test_ols_point_estimates():
    y, X, n_u, n_t = _panel()
    out = beck_katz(y, X, n_u, n_t)
    b = np.asarray(out["beta"])
    assert np.abs(b - np.array([0.3, 1.2])).max() < 0.4


def test_pcse_wider_than_ehw():
    y, X, n_u, n_t = _panel()
    out = beck_katz(y, X, n_u, n_t)
    assert (np.asarray(out["se_pcse"]) > np.asarray(out["se_ehw"]) * 0.9).all()


def test_parks_moves_toward_true():
    y, X, n_u, n_t = _panel()
    fg = parks_fgls(y, X, n_u, n_t)
    b = np.asarray(fg["beta"])
    assert np.abs(b - np.array([0.3, 1.2])).max() < 0.4


def test_sigma_shape():
    y, X, n_u, n_t = _panel()
    out = beck_katz(y, X, n_u, n_t)
    sig = np.asarray(out["sigma"])
    assert sig.shape == (n_u, n_u)
    # off-diagonal positive (equicorrelated truth)
    off = sig[np.triu_indices(n_u, 1)]
    assert off.mean() > 0.3


def test_bench_pcse():
    out = bench_pcse()
    assert out["synthetic_pcse_ehw_ratio"] > 1.0
    assert out["synthetic_coverage_pcse"] > out["synthetic_coverage_ehw"]
