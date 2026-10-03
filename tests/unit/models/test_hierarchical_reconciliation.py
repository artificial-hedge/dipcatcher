"""Tests for MinT hierarchical forecast reconciliation."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.hierarchical_reconciliation import (
    bench_reconciliation,
    incoherence,
    mint_weights,
    reconcile,
)


def _hierarchy(seed: int = 0, n_t: int = 200):
    rng = np.random.default_rng(seed)
    b = rng.standard_normal((n_t, 2))
    a = b.sum(axis=1)
    y = np.stack([a, b[:, 0], b[:, 1]], axis=1)
    s = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    resid = y[:-1] - np.roll(y[:-1], 1, axis=0)
    resid = resid[1:] + 0.1 * rng.standard_normal((n_t - 2, 3))
    return s, y, resid


def test_coherence_exact():
    s, y, resid = _hierarchy()
    yhat = y[-1] + np.array([0.5, 0.0, 0.0])  # aggregate only noise
    for method in ("ols", "wls", "shrink"):
        yr = reconcile(s, yhat, resid, method)
        assert incoherence(s, yr) < 1e-8


def test_bottom_forecasts_unchanged_when_coherent():
    s, y, resid = _hierarchy()
    yhat = y[-1].copy()  # already coherent
    yr = reconcile(s, yhat, resid, "ols")
    assert np.allclose(yr, yhat, atol=1e-8)


def test_mint_weights_shapes():
    s, _, resid = _hierarchy()
    out = mint_weights(s, resid, "shrink")
    assert out["g"].shape == (2, 3)
    assert out["p"].shape == (3, 3)
    # P is a projection onto the coherent subspace: P S = S
    assert np.allclose(out["p"] @ s, s, atol=1e-8)


def test_wls_uses_residual_variance():
    s, _, resid = _hierarchy()
    resid[:, 0] *= 10.0
    w = mint_weights(s, resid, "wls")["w"]
    assert w[0, 0] > w[1, 1]


def test_fail_closed():
    s, y, resid = _hierarchy()
    with pytest.raises(ValueError):
        mint_weights(np.eye(2), resid)
    with pytest.raises(ValueError):
        mint_weights(s, resid[:3])
    with pytest.raises(ValueError):
        mint_weights(s, resid, "bogus")
    with pytest.raises(ValueError):
        reconcile(s, y[-1][:2], resid)
    with pytest.raises(ValueError):
        reconcile(s, np.full(3, np.nan), resid)


def test_bench():
    res = bench_reconciliation()
    assert res["synthetic_score"] == 1.0
    assert res["synthetic_incoherence_post"] < 1e-8
    assert res["synthetic_incoherence_pre"] > 0.0
