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


def test_bench_convex_hull():
    out = bench_convex_hull(seed=20261231)
    assert out["synthetic_hull_contains_all"] == 1.0
    assert out["synthetic_hull_square_area_err"] < 1e-9
    assert all(np.isfinite(v) for v in out.values())
