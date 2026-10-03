"""Tests for probabilistic roadmap planning."""

from __future__ import annotations

import math

import numpy as np

from quant_fund.models.prm import bench_prm, prm_build, prm_query


def test_prm_build_nodes_edges():
    P, adj = prm_build(80, [(0.5, 0.5, 0.2)], k=5, seed=0)
    assert len(P) == 80
    # undirected: every edge appears twice
    for i, nbrs in enumerate(adj):
        for j, c in nbrs:
            assert any(k2 == i for k2, _ in adj[j])
            assert c > 0


def test_prm_query_free_space():
    P, adj = prm_build(200, [], k=8, seed=1)
    cost, expanded = prm_query(P, adj, (0.05, 0.05), (0.95, 0.95), [])
    lb = math.hypot(0.9, 0.9)
    assert math.isfinite(cost)
    assert cost < 1.4 * lb
    assert expanded > 0


def test_prm_query_detour():
    obs = [(0.5, 0.5, 0.15)]
    P, adj = prm_build(300, obs, k=8, seed=2)
    cost, _ = prm_query(P, adj, (0.05, 0.05), (0.95, 0.95), obs)
    lb = math.hypot(0.9, 0.9)
    assert math.isfinite(cost)
    # must detour around the disc → strictly longer
    assert cost > lb + 0.05


def test_bench_prm():
    out = bench_prm(seed=20261231)
    assert out["synthetic_prm_feasible"] == 1.0
    assert out["synthetic_prm_lb_ratio"] >= 1.0
    assert all(np.isfinite(v) for v in out.values())
