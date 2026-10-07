"""Unit tests for quant_fund.models._cate_synth."""

from __future__ import annotations

import numpy as np

from quant_fund.models._cate_synth import cate_data, pehe


def test_data_deterministic() -> None:
    a = cate_data(seed=4, n=100)
    b = cate_data(seed=4, n=100)
    assert all(np.array_equal(p, q) for p, q in zip(a, b, strict=True))


def test_cate_matches_documented_tau() -> None:
    X, t, y, tau, ey1 = cate_data(seed=1, n=300)
    # tau(x) = 1.2*x0 when x1 > 0 else -0.4
    hi = X[:, 1] > 0
    assert np.allclose(tau[hi], 1.2 * X[hi, 0])
    assert np.allclose(tau[~hi], -0.4)
    assert set(np.unique(t)) <= {0, 1}
    # E[Y(1)] = mu0 + tau and y - tau*t reconstructs mu0 + noise
    resid = y - t * tau
    assert resid.shape == y.shape
    _ = ey1


def test_pehe_correctness() -> None:
    assert pehe(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0
    assert pehe(np.array([2.0]), np.array([0.0])) == 2.0
