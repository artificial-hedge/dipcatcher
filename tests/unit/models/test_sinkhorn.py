import numpy as np

from quant_fund.models.sinkhorn import bench_sinkhorn, exact_transport, sinkhorn


def test_sinkhorn_marginals():
    rng = np.random.default_rng(0)
    n = 8
    cost = rng.random((n, n))
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(n))
    p, c = sinkhorn(a, b, cost, eps=0.05)
    assert np.allclose(p.sum(1), a, atol=1e-6)
    assert np.allclose(p.sum(0), b, atol=1e-6)
    assert c >= 0.0


def test_exact_transport_small():
    a = np.array([0.5, 0.5])
    b = np.array([0.5, 0.5])
    cost = np.array([[0.0, 1.0], [1.0, 0.0]])
    p, c = exact_transport(a, b, cost)
    assert abs(c) < 1e-9
    assert np.allclose(p, np.eye(2) * 0.5, atol=1e-6)


def test_bench_fine_gap():
    out = bench_sinkhorn(seed=2)
    assert out["synthetic_fine_gap"] < 0.01
    assert out["synthetic_marginal_err"] < 1e-6
