import numpy as np
from scipy.stats import wasserstein_distance

from quant_fund.models.emd_lp import bench_emd_lp, bures_wasserstein, emd_1d, emd_lp


def test_emd1d_matches_scipy():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(30)
    y = rng.standard_normal(30) + 0.5
    assert abs(emd_1d(x, y) - wasserstein_distance(x, y)) < 1e-10


def test_emd_lp_identical_zero():
    pts = np.random.default_rng(1).random((10, 2))
    assert emd_lp(pts, pts) < 1e-9


def test_bures_closed_form():
    w = bures_wasserstein(np.zeros(2), np.eye(2), np.zeros(2), np.eye(2))
    assert abs(w) < 1e-9
    w2 = bures_wasserstein(np.zeros(2), np.eye(2), np.ones(2), np.eye(2))
    assert abs(w2 - np.sqrt(2)) < 1e-9


def test_bench_err_zero():
    out = bench_emd_lp(seed=3)
    assert out["synthetic_emd1d_err"] < 1e-9
    assert out["synthetic_bures_err"] < 1e-9
