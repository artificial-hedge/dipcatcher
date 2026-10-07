"""Unit tests for quant_fund.models._approx_synth."""

from __future__ import annotations

import numpy as np

from quant_fund.models._approx_synth import (
    G_EDGES,
    G_N,
    KS_CAP,
    KS_V,
    KS_W,
    brute_knapsack,
    brute_maxcut,
    brute_vc,
)


def test_vertex_cover_known_answers() -> None:
    # triangle needs 2; a path of 2 edges needs 1; no edges needs 0
    assert brute_vc([(0, 1), (1, 2), (0, 2)], 3) == 2
    assert brute_vc([(0, 1), (1, 2)], 3) == 1
    assert brute_vc([], 3) == 0
    assert brute_vc(G_EDGES, G_N) <= G_N


def test_maxcut_known_answers() -> None:
    # complete graph K3: cut = 2; bipartite chain: all edges cross
    assert brute_maxcut([(0, 1), (1, 2), (0, 2)], 3) == 2
    assert brute_maxcut([(0, 1), (1, 2)], 3) == 2
    assert brute_maxcut([], 4) == 0


def test_knapsack_brute_force() -> None:
    # single item fits exactly
    assert brute_knapsack(np.array([5.0]), np.array([3.0]), 3.0) == 5.0
    # item too heavy -> nothing taken
    assert brute_knapsack(np.array([5.0]), np.array([4.0]), 3.0) == 0.0
    # empty inventory
    assert brute_knapsack(np.array([]), np.array([]), 3.0) == 0.0
    v = brute_knapsack(KS_V, KS_W, KS_CAP)
    assert v > 0.0
    # fixture optimum: items {0,3,6} -> 13+8+26=47 value, weight 4+3+8=15;
    # better is {0,3,4,7}? check optimality lies within capacity
    assert v >= 47.0


def test_fixture_dimensions_agree() -> None:
    assert len(KS_V) == len(KS_W)
    assert all(0 <= u < G_N and 0 <= v < G_N for u, v in G_EDGES)
