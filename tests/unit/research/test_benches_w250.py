"""Wave-250 adapter tests."""

from quant_fund.research.benches_w250 import (
    bench_astar_search_family,
    bench_bidirectional_dijkstra_family,
    bench_bron_kerbosch_family,
    bench_critical_path_family,
    bench_dinic_flow_family,
    bench_mincost_flow_family,
)


def test_bench_astar_search_family():
    out = bench_astar_search_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_bidirectional_dijkstra_family():
    out = bench_bidirectional_dijkstra_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_bron_kerbosch_family():
    out = bench_bron_kerbosch_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_critical_path_family():
    out = bench_critical_path_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_dinic_flow_family():
    out = bench_dinic_flow_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_mincost_flow_family():
    out = bench_mincost_flow_family()
    assert out and all(k.startswith("synthetic_") for k in out)
