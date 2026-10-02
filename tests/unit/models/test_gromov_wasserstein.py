import numpy as np

from quant_fund.models.gromov_wasserstein import bench_gromov_wasserstein, gromov_wasserstein


def test_gw_isomorphic_small():
    rng = np.random.default_rng(0)
    n = 6
    x = rng.random((n, 2))
    c1 = np.sqrt(((x[:, None] - x[None, :]) ** 2).sum(-1))
    perm = rng.permutation(n)
    c2 = c1[np.ix_(perm, perm)]
    p = np.full(n, 1.0 / n)
    t, gw = gromov_wasserstein(c1, c2, p, p, eps=0.01, max_iter=30)
    assert np.allclose(t.sum(1), p, atol=1e-3)
    assert gw >= 0.0


def test_bench_sep():
    out = bench_gromov_wasserstein(seed=4)
    assert out["synthetic_gw_diff"] > out["synthetic_gw_iso"]
