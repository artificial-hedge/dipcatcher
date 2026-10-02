import numpy as np

from quant_fund.models.shortest_paths import (
    a_star,
    bellman_ford,
    bench_shortest_paths,
    dijkstra,
    floyd_warshall,
)


def test_dijkstra_vs_fw():
    rng = np.random.default_rng(0)
    n = 30
    edges = [(int(u), int(v), 1.0) for u, v in rng.integers(0, n, (100, 2))]
    d = dijkstra(n, edges, 0)
    fw = floyd_warshall(n, edges)
    fin = np.isfinite(fw[0])
    assert np.allclose(d[fin], fw[0][fin])


def test_bellman_negative_edges():
    edges = [(0, 1, 2.0), (1, 2, -3.0), (0, 2, 1.0)]
    d, neg = bellman_ford(3, edges, 0)
    assert d[2] == -1.0
    assert not neg


def test_bellman_detects_cycle():
    edges = [(0, 1, -1.0), (1, 2, -1.0), (2, 0, -1.0)]
    _, neg = bellman_ford(3, edges, 0)
    assert neg


def test_a_star_optimal():
    edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 5.0)]
    d, path = a_star(3, edges, 0, 2, lambda u: 0.0)
    assert d == 2.0
    assert path == [0, 1, 2]


def test_floyd_warshall():
    edges = [(0, 1, 1.0), (1, 2, 2.0)]
    d = floyd_warshall(3, edges)
    assert d[0, 2] == 3.0


def test_bench_keys():
    out = bench_shortest_paths()
    assert out["synthetic_dijkstra_err"] < 1e-9
    assert out["synthetic_bellman_err"] < 1e-9
