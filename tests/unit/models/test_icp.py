"""Tests for iterative closest point."""

from __future__ import annotations

import numpy as np

from quant_fund.models.icp import bench_icp, icp


def test_icp_aligned_is_identity():
    rng = np.random.default_rng(11)
    P = rng.normal(size=(80, 3))
    R, t, rms, iters = icp(P, P, max_iter=20)
    assert rms < 1e-8


def test_icp_small_shift():
    rng = np.random.default_rng(12)
    th = rng.uniform(0, 2 * np.pi, 150)
    P = np.stack([np.cos(th), np.sin(th), np.zeros(150)], axis=1)
    Q = P + np.array([0.05, -0.02, 0.0])
    R, t, rms, iters = icp(P, Q, max_iter=60)
    assert rms < 0.01


def test_bench_icp():
    out = bench_icp(seed=20261231)
    assert out["synthetic_icp_converged"] == 1.0
    assert out["synthetic_icp_rms"] < 0.05
    assert all(np.isfinite(v) for v in out.values())
