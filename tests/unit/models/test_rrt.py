"""Tests for RRT/RRT* planning."""

from __future__ import annotations

import math

import numpy as np

from quant_fund.models.rrt import bench_rrt, rrt


def test_rrt_free_space_straight():
    L, path, iters = rrt((0.1, 0.1), (0.9, 0.9), [], seed=1, max_iter=2000)
    assert math.isfinite(L)
    # path length should be close to the straight-line distance
    assert 1.35 * math.hypot(0.8, 0.8) > L


def test_rrt_avoids_obstacles():
    obstacles = [(0.5, 0.5, 0.12)]
    L, path, _ = rrt((0.05, 0.05), (0.95, 0.95), obstacles, seed=3, max_iter=4000)
    assert math.isfinite(L)
    ox, oy, r = obstacles[0]
    for x, y in path:
        assert math.hypot(x - ox, y - oy) >= r - 1e-9


def test_rrtstar_improves():
    obstacles = [(0.5, 0.5, 0.12)]
    L_rrt, _, _ = rrt((0.05, 0.05), (0.95, 0.95), obstacles, seed=5, max_iter=3000)
    L_star, _, _ = rrt(
        (0.05, 0.05),
        (0.95, 0.95),
        obstacles,
        seed=5,
        max_iter=3000,
        star=True,
    )
    assert L_star <= L_rrt + 1e-9


def test_bench_rrt():
    out = bench_rrt(seed=20261231)
    assert out["synthetic_rrt_feasible_rate"] >= 0.75
    assert out["synthetic_rrtstar_cost"] <= out["synthetic_rrt_min_cost"] + 1e-9
    assert all(np.isfinite(v) for v in out.values())
