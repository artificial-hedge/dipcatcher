import numpy as np

from quant_fund.models.randomized_svd import bench_randomized_svd, randomized_svd


def test_rsvd_shapes():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((80, 60))
    u, s, vt = randomized_svd(a, 5, rng)
    assert u.shape == (80, 5) and s.shape == (5,) and vt.shape == (5, 60)


def test_rsvd_low_rank():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((100, 8)) @ rng.standard_normal((8, 90))
    u, s, vt = randomized_svd(a, 8, rng, power=2)
    err = np.linalg.norm(a - u @ np.diag(s) @ vt) / np.linalg.norm(a)
    assert err < 1e-8


def test_bench_gap_small():
    out = bench_randomized_svd(seed=4)
    assert out["synthetic_gap"] < 0.01
    assert out["synthetic_sv_err"] < 0.01
