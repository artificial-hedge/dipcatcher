"""Adversarial probes for a_star_route (SYNTHETIC)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.a_star_route import _bfs, bench_a_star_route, route


def test_route_matches_bfs_on_random_grids():
    rng = np.random.default_rng(0)
    for _ in range(30):
        g = (rng.random((8, 8)) < 0.25).astype(int).tolist()
        g[0][0] = g[7][7] = 0
        assert route(g, (0, 0), (7, 7)) == _bfs(g, (0, 0), (7, 7))


def test_route_known_answers():
    assert route([[0] * 3 for _ in range(3)], (0, 0), (2, 2)) == 4
    g = [[0, 1, 0], [1, 1, 0], [0, 0, 0]]
    assert route(g, (0, 0), (0, 2)) == -1  # walled off
    assert route([[0]], (0, 0), (0, 0)) == 0


def test_route_hostile():
    g = [[0, 0], [0, 0]]
    with pytest.raises(ValueError):
        route(g, (-1, 0), (1, 1))
    with pytest.raises(ValueError):
        route(g, (0, 0), (0, 5))
    with pytest.raises(ValueError):
        route([[0, 1]], (0, 1), (0, 0))  # src blocked
    with pytest.raises(ValueError):
        route([[0, 1]], (0, 0), (0, 1))  # dst blocked
    with pytest.raises(ValueError):
        route([], (0, 0), (0, 0))
    with pytest.raises(ValueError):
        route([[0, 0], [0]], (0, 0), (0, 0))  # ragged


def test_route_manhattan_lower_bound():
    rng = np.random.default_rng(1)
    for _ in range(20):
        g = (rng.random((7, 7)) < 0.2).astype(int).tolist()
        g[0][0] = g[6][6] = 0
        d = route(g, (0, 0), (6, 6))
        assert d == -1 or d >= 12  # never shorter than Manhattan


def test_bench_route_full_pass():
    assert bench_a_star_route()["synthetic_route"] == 1.0
