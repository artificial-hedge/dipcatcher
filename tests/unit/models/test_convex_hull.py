"""Tests for Graham-scan convex hull."""

from __future__ import annotations

import numpy as np

from quant_fund.models.convex_hull import (
    bench_convex_hull,
    convex_hull,
    hull_area,
    point_in_hull,
)


def test_hull_square():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]])
    h = convex_hull(sq)
    assert len(h) == 4
    assert abs(hull_area(h) - 1.0) < 1e-9


def test_hull_contains():
    rng = np.random.default_rng(8)
    P = rng.normal(size=(100, 2))
    h = convex_hull(P)
    assert all(point_in_hull(p, h, tol=1e-6) for p in P)


def test_collinear_anchor_ray_keeps_extreme() -> None:
    # Two points on the minimum-angle ray from the anchor: the farthest is
    # the true vertex. A farthest-first same-angle order let the nearer
    # point replace it, dropping (2,0) off the hull entirely.
    p = np.array([[0, 0], [1, 0], [2, 0], [0, 1]], dtype=float)
    h = convex_hull(p)
    assert len(h) == 3
    assert any(np.allclose(row, [2.0, 0.0]) for row in h)
    assert all(point_in_hull(q, h, tol=1e-9) for q in p)


def test_collinear_mid_ray_keeps_extreme() -> None:
    # Same defect on an interior (non-first) ray: (2,2) collinear with the
    # anchor and (1,1); the farther point must survive.
    p = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 2], [3, 0]], dtype=float)
    h = convex_hull(p)
    assert any(np.allclose(row, [2.0, 2.0]) for row in h)
    assert all(point_in_hull(q, h, tol=1e-9) for q in p)


def test_bench_convex_hull():
    out = bench_convex_hull(seed=20261231)
    assert out["synthetic_hull_contains_all"] == 1.0
    assert out["synthetic_hull_square_area_err"] < 1e-9
    assert all(np.isfinite(v) for v in out.values())
