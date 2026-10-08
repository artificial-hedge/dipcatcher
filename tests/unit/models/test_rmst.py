"""Tests for rmst — Royston-Parmar restricted mean survival time."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.rmst import bench_rmst, rmst, rmst_compare


def _exp(lam: float, n: int, seed: int):
    rng = np.random.default_rng(seed)
    return rng.exponential(1.0 / lam, size=n), np.ones(n)


def test_rmst_recovers_exponential():
    t, d = _exp(0.5, 600, 0)
    out = rmst(t, d, 2.0)
    true = (1.0 - np.exp(-0.5 * 2.0)) / 0.5
    assert abs(out["rmst"] - true) / true < 0.10
    assert out["se"] > 0


def test_rmst_increases_with_tau():
    t, d = _exp(0.5, 400, 1)
    a = rmst(t, d, 1.0)["rmst"]
    b = rmst(t, d, 3.0)["rmst"]
    assert b > a


def test_compare_detects_longer_arm():
    t1, d1 = _exp(0.5, 500, 2)
    t2, d2 = _exp(1.0, 500, 3)
    out = rmst_compare(t1, d1, t2, d2, 2.0)
    assert out["rmst2"] < out["rmst1"]
    assert out["diff"] > 0
    assert out["p_diff"] < 0.05


def test_identical_arms_insignificant():
    t1, d1 = _exp(0.5, 500, 4)
    t2, d2 = _exp(0.5, 500, 5)
    out = rmst_compare(t1, d1, t2, d2, 2.0)
    assert out["p_diff"] > 0.01


def test_bad_inputs():
    with pytest.raises(ValueError):
        rmst(np.array([1.0]), np.array([1.0]), 1.0)
    with pytest.raises(ValueError):
        rmst(np.array([1.0, -2.0, 3.0]), np.array([1.0, 1.0, 1.0]), 1.0)


def test_bench():
    assert bench_rmst()["synthetic_score"] == 1.0
