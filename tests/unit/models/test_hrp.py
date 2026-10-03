"""Tests for hrp — hierarchical risk parity weights."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.hrp import bench_hrp, hrp_weights


def test_weights_sum_to_one_nonnegative():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((200, 6))
    cov = np.cov(a.T) + 0.1 * np.eye(6)
    w = hrp_weights(cov)
    assert w.shape == (6,)
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-9)
    assert (w >= -1e-12).all()


def test_diagonal_cov_gives_ivp_like():
    dv = np.diag(np.array([1.0, 4.0, 9.0]))
    w = hrp_weights(dv)
    ivp = 1.0 / np.diag(dv)
    ivp /= ivp.sum()
    assert np.abs(np.sort(w) - np.sort(ivp)).max() < 0.2


def test_block_corr_balanced():
    corr = np.full((6, 6), 0.05)
    corr[:3, :3] = 0.9
    corr[3:, 3:] = 0.9
    np.fill_diagonal(corr, 1.0)
    cov = corr.copy()
    w = hrp_weights(cov)
    # each block should carry roughly half the total weight
    assert abs(w[:3].sum() - w[3:].sum()) < 0.2


def test_fail_closed_nonsquare():
    with pytest.raises(ValueError):
        hrp_weights(np.ones((3, 4)))


def test_fail_closed_asymmetric():
    c = np.eye(3)
    c[0, 1] = 0.5
    with pytest.raises(ValueError):
        hrp_weights(c)


def test_bench():
    out = bench_hrp()
    assert out["score"] == 1.0
