"""Tests for Kabsch rigid alignment."""

from __future__ import annotations

import numpy as np

from quant_fund.models.kabsch import bench_kabsch, kabsch


def test_kabsch_recovers_rigid():
    rng = np.random.default_rng(5)
    P = rng.normal(size=(50, 3))
    ang = 0.4
    R_true = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1.0]])
    Q = P @ R_true + np.array([1.0, 2.0, -1.0])
    R, t, s, rms = kabsch(P, Q)
    assert rms < 1e-10
    assert np.abs(R - R_true).max() < 1e-10
    assert abs(np.linalg.det(R) - 1.0) < 1e-9


def test_kabsch_rejects_reflection():
    rng = np.random.default_rng(6)
    P = rng.normal(size=(40, 3))
    Q = P.copy()
    Q[:, 0] = -Q[:, 0]  # mirror — not a rotation
    R, t, s, rms = kabsch(P, Q)
    assert np.linalg.det(R) > 0  # still a proper rotation


def test_bench_kabsch():
    out = bench_kabsch(seed=20261231)
    assert out["synthetic_kabsch_rms"] < 1e-10
    assert out["synthetic_kabsch_scale_err"] < 1e-6
    assert all(np.isfinite(v) for v in out.values())
