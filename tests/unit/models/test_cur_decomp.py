import numpy as np

from quant_fund.models.cur_decomp import bench_cur_decomp, cur_decomp, leverage_scores


def test_leverage_scores_sum():
    a = np.random.default_rng(0).standard_normal((50, 40))
    lev = leverage_scores(a, 5)
    assert abs(lev.sum() - 1.0) < 1e-9
    assert lev.shape == (40,)


def test_cur_shapes():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((60, 50))
    c, u, r = cur_decomp(a, 5, rng, n_cols=20, n_rows=20)
    assert c.shape == (60, 20) and r.shape == (20, 50)


def test_bench_near_opt():
    out = bench_cur_decomp(seed=6)
    assert out["synthetic_ratio"] < 1.5
