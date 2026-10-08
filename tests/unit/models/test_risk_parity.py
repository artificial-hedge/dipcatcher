"""Tests for risk_parity — ERC portfolios."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.risk_parity import (
    bench_risk_parity,
    concentrated_parity,
    erc_weights,
)


def _cov(seed: int = 0, n: int = 4):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    c = a @ a.T
    d = np.diag(1.0 / np.sqrt(np.diag(c)))
    r = d @ c @ d
    vols = np.linspace(0.05, 0.30, n)
    return np.diag(vols) @ r @ np.diag(vols)


def test_rc_shares_equal():
    cov = _cov()
    out = erc_weights(cov)
    share = np.asarray(out["rc_share"])
    assert np.abs(share - 0.25).max() < 0.01


def test_weights_sum_one():
    cov = _cov(seed=1)
    out = erc_weights(cov)
    w = np.asarray(out["w"])
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w > 0)


def test_low_vol_gets_high_weight():
    # diagonal variances: ERC -> w ~ 1/sigma_i
    cov = np.diag([0.05, 0.10, 0.20, 0.40])
    out = erc_weights(cov)
    w = np.asarray(out["w"])
    assert np.argmax(w) == 0
    assert w[0] > w[3] * 2.5  # ratio ~ sqrt(0.40/0.05) = 2.83


def test_budgeted_erc():
    cov = _cov(seed=2, n=3)
    out = erc_weights(cov, budget=np.array([0.5, 0.3, 0.2]))
    share = np.asarray(out["rc_share"])
    assert np.abs(share - np.array([0.5, 0.3, 0.2])).max() < 0.02


def test_naive_parity_baseline():
    cov = _cov(seed=3)
    out = concentrated_parity(cov)
    w = np.asarray(out["w"])
    assert w.sum() == pytest.approx(1.0)


def test_fail_closed_singular():
    cov = np.ones((3, 3))  # rank-1, singular
    cov[np.diag_indices(3)] = 1.0
    with pytest.raises(ValueError):
        erc_weights(cov)


def test_fail_closed_asymmetric():
    cov = np.array([[1.0, 0.5], [0.1, 1.0]])
    with pytest.raises(ValueError):
        erc_weights(cov)


def test_bench():
    out = bench_risk_parity()
    assert out["synthetic_score"] == 1.0
