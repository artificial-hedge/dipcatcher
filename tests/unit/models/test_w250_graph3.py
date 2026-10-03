"""Wave-250 graph-3 canon tests."""

from quant_fund.models.astar_search import astar, bench_astar_search
from quant_fund.models.bidirectional_dijkstra import (
    bench_bidirectional_dijkstra,
    bidirectional_dijkstra,
)
from quant_fund.models.bron_kerbosch import bench_bron_kerbosch, bron_kerbosch
from quant_fund.models.critical_path import bench_critical_path, cpm
from quant_fund.models.dinic_flow import bench_dinic_flow, dinic
from quant_fund.models.mincost_flow import bench_mincost_flow, mincost_flow


def test_dinic_basic():
    f = dinic(4, [(0, 1, 3), (0, 2, 2), (1, 3, 2), (2, 3, 4)], 0, 3)
    assert f == 4


def test_dinic_bench():
    assert bench_dinic_flow()["synthetic_maxflow_mincut"] == 1.0


def test_mincost_basic():
    f, c = mincost_flow(3, [(0, 1, 2, 1), (1, 2, 2, 1), (0, 2, 1, 5)], 0, 2, 2)
    assert f == 2 and c == 4


def test_mincost_bench():
    assert bench_mincost_flow()["synthetic_cost_exact"] == 1.0


def test_astar_basic():
    edges = {0: [(1, 1.0), (2, 5.0)], 1: [(2, 1.0)]}
    d, _ = astar(3, edges, {0: 0, 1: 1, 2: 0}, 0, 2)
    assert d == 2.0


def test_astar_bench():
    assert bench_astar_search()["synthetic_optimal_vs_dijkstra"] == 1.0


def test_bidir_basic():
    ef = {0: [(1, 1.0)], 1: [(2, 1.0)]}
    eb = {1: [(0, 1.0)], 2: [(1, 1.0)]}
    assert bidirectional_dijkstra(ef, eb, 0, 2) == 2.0


def test_bidir_bench():
    assert bench_bidirectional_dijkstra()["synthetic_distance_exact"] == 1.0


def test_cpm_basic():
    length, tf, path = cpm(3, [(0, 1, 2.0), (1, 2, 3.0), (0, 2, 4.0)])
    assert length == 5.0


def test_cpm_bench():
    assert bench_critical_path()["synthetic_longest_exact"] == 1.0


def test_bk_basic():
    adj = {0: {1, 2}, 1: {0, 2}, 2: {0, 1}, 3: set()}
    cls = bron_kerbosch(adj)
    assert frozenset({0, 1, 2}) in {frozenset(c) for c in cls}


def test_bk_bench():
    assert bench_bron_kerbosch()["synthetic_maximal_cliques_exact"] == 1.0
