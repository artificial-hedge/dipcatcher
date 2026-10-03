import pytest

from quant_fund.models.graph_traversal import (
    bench_graph_traversal,
    bfs,
    dfs_iter,
    is_bipartite,
    kahn_topo,
)


def test_bfs_levels():
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    r = bfs(4, edges, 0)
    assert r["level"][0] == 0
    assert r["level"][3] == 2


def test_dfs_visits_all():
    edges = [(i, i + 1) for i in range(9)]
    assert len(dfs_iter(10, edges, 0)) == 10


def test_topo_valid():
    edges = [(0, 2), (1, 2), (1, 3), (3, 4)]
    t = kahn_topo(5, edges)
    pos = {v: i for i, v in enumerate(t)}
    for u, v in edges:
        assert pos[u] < pos[v]


def test_topo_cycle_raises():
    with pytest.raises(ValueError):
        kahn_topo(3, [(0, 1), (1, 2), (2, 0)])


def test_bipartite():
    ok, _ = is_bipartite(4, [(0, 1), (1, 2), (2, 3)])
    assert ok
    bad, _ = is_bipartite(3, [(0, 1), (1, 2), (2, 0)])
    assert not bad


def test_bench_keys():
    out = bench_graph_traversal()
    assert out["synthetic_topo_valid"] == 1.0
    assert out["synthetic_bip_neg"] == 1.0
