"""Tests for procrustes — orthogonal alignment + GPA."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.procrustes import (
    bench_procrustes,
    generalized_procrustes,
    procrustes,
)


def test_recovers_rotation_and_scale():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((30, 2))
    ang = 0.5
    r = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    y = 3.0 * (x @ r) + np.array([5.0, -2.0])
    out = procrustes(x, y)
    assert np.abs(np.asarray(out["rotation"]) - r.T).max() < 1e-6
    # scale maps target->ref: y was inflated 3x so c* = 1/3
    assert float(out["scale"]) == pytest.approx(1.0 / 3.0, rel=1e-6)
    assert float(out["disparity"]) < 1e-10


def test_aligned_shape_matches():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((25, 3))
    y = 1.7 * x + 0.001 * rng.standard_normal((25, 3))
    out = procrustes(x, y)
    aligned = np.asarray(out["aligned"])
    assert np.abs(aligned - x).max() < 0.01


def test_gpa_converges():
    rng = np.random.default_rng(3)
    base = rng.standard_normal((20, 2))
    shapes = [base + 0.01 * rng.standard_normal((20, 2)) for _ in range(3)]
    out = generalized_procrustes(shapes)
    assert float(out["dispersion"]) < 0.01
    assert np.asarray(out["mean_shape"]).shape == (20, 2)


def test_fail_closed_mismatch():
    with pytest.raises(ValueError):
        procrustes(np.ones((5, 2)), np.ones((6, 2)))


def test_fail_closed_degenerate():
    with pytest.raises(ValueError):
        procrustes(np.ones((5, 2)), np.ones((5, 2)) * 2)


def test_bench():
    out = bench_procrustes()
    assert out["score"] == 1.0
