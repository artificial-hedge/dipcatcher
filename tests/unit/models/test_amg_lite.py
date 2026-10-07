"""Tests for models/amg_lite.py — strength matrix, C/F split, interpolation."""

from __future__ import annotations

import numpy as np

from quant_fund.models.amg_lite import (
    bench_amg_lite,
    interp_matrix,
    poisson_2d,
    select_coarse,
    strength_matrix,
    two_level_amg,
)


def test_strength_matrix_threshold() -> None:
    a = np.array([[4.0, -2.0, 0.0], [-2.0, 4.0, -1.0], [0.0, -1.0, 4.0]])
    s = strength_matrix(a, theta=0.4)
    assert s[0].tolist() == [False, True, False]
    assert s[1].tolist() == [True, False, True]
    assert s[2].tolist() == [False, True, False]


def test_interp_fallback_uses_real_connection() -> None:
    """Fine node whose only off-diagonal edge is to another F node: the
    |a_ij| argmax must find that edge (not a zero entry inflated by a
    large diagonal), then fall back to the nearest C node."""
    a = np.array(
        [
            [4.0, -1.0, 0.0, 0.0],
            [-1.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ]
    )
    a[2, 1] = a[1, 2] = 0.5  # weak real edge node 2 <-> node 1
    s = np.zeros((4, 4), dtype=bool)
    s[2, 1] = True  # strong but node 1 is fine
    is_c = np.array([True, False, False, True])
    p = interp_matrix(a, s, is_c)
    # node 2 must interpolate from the nearest COARSE node (3), never
    # from node 0 whose coupling a[2,0] is exactly zero
    assert p[2, 0] == 0.0
    assert p[2, 1] == 1.0


def test_two_level_matches_dense() -> None:
    rng = np.random.default_rng(0)
    a = poisson_2d(6)
    b = rng.normal(size=36)
    exact = np.linalg.solve(a, b)
    x = np.zeros(36)
    for _ in range(30):
        x = two_level_amg(a, b, x)
    assert np.linalg.norm(x - exact) < 1e-6 * np.linalg.norm(exact)


def test_select_coarse_partition() -> None:
    a = poisson_2d(4)
    is_c = select_coarse(strength_matrix(a))
    assert 0.0 < is_c.mean() < 1.0


def test_bench_amg_lite() -> None:
    out = bench_amg_lite()
    assert out["synthetic_amg_lite"] >= 0.75
