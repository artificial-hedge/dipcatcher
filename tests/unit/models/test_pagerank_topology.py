import numpy as np

from quant_fund.models.pagerank_topology import (
    bench_pagerank,
    conductance,
    hits,
    pagerank,
    spectral_bisect,
)


def test_pagerank_sums_to_one():
    adj = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    pr = pagerank(adj)
    np.testing.assert_allclose(pr.sum(), 1.0, atol=1e-9)
    np.testing.assert_allclose(pr, 1.0 / 3, atol=1e-3)


def test_pagerank_personalized():
    adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    pers = np.array([1.0, 0, 0])
    pr = pagerank(adj, pers=pers)
    pr_unif = pagerank(adj)
    assert pr[0] > pr_unif[0]


def test_hits_star_authority():
    adj = np.zeros((5, 5))
    adj[1:, 0] = 1
    h = hits(adj)
    assert int(np.argmax(np.asarray(h["authority"]))) == 0


def test_conductance_bounds():
    adj = np.ones((4, 4)) - np.eye(4)
    s = np.array([True, True, False, False])
    phi = conductance(adj, s)
    assert 0 <= phi <= 1


def test_spectral_bisect_clusters():
    adj = np.zeros((6, 6))
    adj[np.ix_([0, 1, 2], [0, 1, 2])] = 1
    adj[np.ix_([3, 4, 5], [3, 4, 5])] = 1
    np.fill_diagonal(adj, 0)
    part = spectral_bisect(adj)
    assert part[0] == part[1] == part[2]
    assert part[0] != part[3]


def test_bench_pagerank_runs():
    out = bench_pagerank(seed=544)
    assert out["synthetic_pr_hub_mass"] > 0
    assert out["synthetic_conductance_comm"] < out["synthetic_conductance_rand"]
