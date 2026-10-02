"""Tests for mst_topology — MST/PMFG market networks."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.mst_topology import bench_mst_topology, mst_build, pmfg_build


def _factor_corr(seed: int = 0, n: int = 8, t: int = 300):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=t)
    lams = np.linspace(0.9, 0.4, n)
    x = np.column_stack(
        [lams[j] * f + np.sqrt(1 - lams[j] ** 2) * rng.normal(size=t) for j in range(n)]
    )
    return np.asarray(np.corrcoef(x, rowvar=False), dtype=np.float64)


def test_mst_edge_count():
    c = _factor_corr()
    out = mst_build(c)
    edges = np.asarray(out["edges"])
    assert edges.shape[0] == c.shape[0] - 1


def test_hub_is_strongest_factor_node():
    c = _factor_corr()
    out = mst_build(c)
    # node 0 has the highest loading -> should be hub most seeds
    assert out["max_degree"] >= 3


def test_star_topology_concentration():
    # pure one-factor: hub gets n-1 or near
    rng = np.random.default_rng(2)
    f = rng.normal(size=500)
    x = np.column_stack([0.95 * f + 0.31 * rng.normal(size=500) for _ in range(6)])
    c = np.asarray(np.corrcoef(x, rowvar=False), dtype=np.float64)
    out = mst_build(c)
    assert out["max_degree"] >= 4


def test_pmfg_dense_enough():
    c = _factor_corr()
    out = pmfg_build(c)
    n = c.shape[0]
    assert out["n_edges"] >= n - 1
    assert out["n_edges"] <= 3 * n - 6


def test_degree_nonnegative():
    c = _factor_corr()
    out = mst_build(c)
    deg = np.asarray(out["degree"])
    assert np.all(deg >= 1)


def test_fail_closed_not_corr():
    bad = np.array([[1.0, 0.5], [0.5, 1.0]])  # k<3
    with pytest.raises(ValueError):
        mst_build(bad)


def test_fail_closed_nonunit_diag():
    bad = np.eye(4) * 0.5
    with pytest.raises(ValueError):
        mst_build(bad)


def test_bench():
    out = bench_mst_topology()
    assert out["score"] == 1.0
