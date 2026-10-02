import numpy as np

from quant_fund.models.unbalanced_ot import bench_unbalanced_ot, unbalanced_sinkhorn


def test_unbalanced_creates_mass():
    rng = np.random.default_rng(0)
    n = 10
    cost = rng.random((n, n))
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(n)) * 1.5
    p, c = unbalanced_sinkhorn(a, b, cost, eps=0.05, rho=0.5)
    assert p.shape == (n, n)
    assert c >= 0.0
    # mass not forced to 1
    assert 0.0 < p.sum() < 2.0


def test_balanced_limit():
    rng = np.random.default_rng(1)
    n = 8
    cost = rng.random((n, n))
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(n))
    p, _ = unbalanced_sinkhorn(a, b, cost, eps=0.05, rho=50.0)
    assert np.allclose(p.sum(1), a, atol=0.02)


def test_bench_keys():
    out = bench_unbalanced_ot(seed=5)
    assert out["synthetic_mass_total"] > 0.0
