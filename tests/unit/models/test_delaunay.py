"""Tests for Bowyer–Watson Delaunay triangulation."""

from __future__ import annotations

import numpy as np

from quant_fund.models.delaunay import bench_delaunay, delaunay


def test_delaunay_grid():
    g = np.array([[i, j] for i in range(3) for j in range(3)], dtype=np.float64)
    tris = delaunay(g)
    assert len(tris) == 8


def test_delaunay_empty_circle():
    from quant_fund.models.delaunay import _circum_contains

    rng = np.random.default_rng(3)
    P = rng.uniform(0, 1, (30, 2))
    tris = delaunay(P)
    for t in tris:
        a, b, c = P[t[0]], P[t[1]], P[t[2]]
        for i, p in enumerate(P):
            if i not in t:
                assert not _circum_contains(a, b, c, p)


def test_bench_delaunay():
    out = bench_delaunay(seed=20261231)
    assert out["synthetic_delaunay_empty_viol"] == 0.0
    assert out["synthetic_delaunay_agree"] == out["synthetic_delaunay_scipy_n"]
    assert all(np.isfinite(v) for v in out.values())
