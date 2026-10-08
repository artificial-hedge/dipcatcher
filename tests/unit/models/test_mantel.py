"""Tests for mantel — distance-matrix correlation."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.mantel import bench_mantel, mantel_partial, mantel_test


def _dist(pts):
    return np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)


def test_identical_matrices_r_one():
    rng = np.random.default_rng(0)
    d = _dist(rng.random((25, 2)))
    out = mantel_test(d, d.copy(), n_perm=99, seed=0)
    assert out["r"] == pytest.approx(1.0)


def test_dependent_detected_independent_not():
    rng = np.random.default_rng(1)
    d = _dist(rng.random((25, 2)))
    d2 = d + 0.1 * rng.standard_normal((25, 25))
    d2 = (d2 + d2.T) / 2
    np.fill_diagonal(d2, 0)
    assert mantel_test(d, d2, n_perm=199, seed=0)["p"] < 0.02
    d3 = _dist(rng.random((25, 2)))
    assert mantel_test(d, d3, n_perm=199, seed=1)["p"] > 0.005


def test_partial_runs():
    rng = np.random.default_rng(2)
    d = _dist(rng.random((20, 2)))
    d2 = _dist(rng.random((20, 2)))
    d3 = _dist(rng.random((20, 2)))
    out = mantel_partial(d, d2, d3, n_perm=99, seed=0)
    assert np.isfinite(out["r_partial"])


def test_fail_closed_nonsquare():
    with pytest.raises(ValueError):
        mantel_test(np.ones((4, 5)), np.ones((4, 5)))


def test_bench():
    out = bench_mantel()
    assert out["synthetic_score"] == 1.0
