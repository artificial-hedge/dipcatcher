import numpy as np

from quant_fund.models.network_flow import (
    bench_network_flow,
    dinic_maxflow,
    min_cut,
)


def test_simple_flow():
    edges = [(0, 1, 3.0), (0, 2, 2.0), (1, 3, 2.0), (2, 3, 4.0)]
    val, flow = dinic_maxflow(4, edges, 0, 3)
    assert abs(val - 4.0) < 1e-9


def test_bottleneck():
    edges = [(0, 1, 10.0), (1, 2, 1.0), (2, 3, 10.0)]
    val, _ = dinic_maxflow(4, edges, 0, 3)
    assert abs(val - 1.0) < 1e-9


def test_duality():
    rng = np.random.default_rng(0)
    n = 20
    edges = [
        (int(u), int(v), float(rng.random() * 8 + 1))
        for u, v in rng.integers(0, n, (60, 2))
        if u != v
    ]
    val, flow = dinic_maxflow(n, edges, 0, n - 1)
    _, cut = min_cut(n, edges, 0, flow)
    assert abs(val - cut) < 1e-9


def test_bench_keys():
    out = bench_network_flow()
    assert out["synthetic_duality_err"] < 1e-9
    assert out["synthetic_conserve_err"] < 1e-9
