"""Tests for minimum-snap trajectory generation."""

from __future__ import annotations

import numpy as np

from quant_fund.models.min_snap import bench_min_snap, eval_traj, min_snap_traj


def test_min_snap_hits_waypoints():
    w = np.array([0.0, 1.0, -0.5, 0.3])
    t = np.array([0.0, 1.0, 2.5, 4.0])
    coefs = min_snap_traj(w, t)
    for tt, ww in zip(t, w, strict=True):
        assert abs(eval_traj(coefs, t, tt) - ww) < 1e-6


def test_min_snap_continuity():
    w = np.array([0.0, 1.0, 0.0])
    t = np.array([0.0, 1.0, 2.0])
    coefs = min_snap_traj(w, t)
    for d in (1, 2, 3):
        dv = abs(eval_traj(coefs, t, 1.0 - 1e-5, d) - eval_traj(coefs, t, 1.0 + 1e-5, d))
        assert dv < 0.02


def test_min_snap_zero_boundary():
    w = np.array([0.2, 0.8, 0.2])
    t = np.array([0.0, 1.0, 2.0])
    coefs = min_snap_traj(w, t)
    assert abs(eval_traj(coefs, t, 0.0, 1)) < 1e-6
    assert abs(eval_traj(coefs, t, 2.0, 1)) < 1e-6


def test_bench_min_snap():
    out = bench_min_snap(seed=20261231)
    assert out["synthetic_minsnap_wp_err"] < 1e-6
    assert out["synthetic_minsnap_cost"] > 0
    assert all(np.isfinite(v) for v in out.values())
