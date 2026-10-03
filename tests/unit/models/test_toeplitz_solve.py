import numpy as np

from quant_fund.models.toeplitz_solve import (
    bench_toeplitz_solve,
    levinson,
    toeplitz_mat,
    yule_walker,
)


def test_levinson_vs_dense():
    rng = np.random.default_rng(0)
    rho = 0.5
    r = rho ** np.arange(8)
    T = toeplitz_mat(r)
    b = rng.standard_normal(8)
    x = levinson(r, b)
    assert np.linalg.norm(x - np.linalg.solve(T, b)) < 1e-10


def test_yule_walker_ar1():
    rho = 0.7
    r = rho ** np.arange(6)
    a = yule_walker(r, 5)
    assert abs(a[0] - rho) < 1e-10
    assert np.abs(a[1:]).max() < 1e-8


def test_residual():
    rng = np.random.default_rng(3)
    r = 0.4 ** np.arange(10)
    b = rng.standard_normal(10)
    T = toeplitz_mat(r)
    x = levinson(r, b)
    assert np.linalg.norm(T @ x - b) / np.linalg.norm(b) < 1e-10


def test_bench():
    out = bench_toeplitz_solve(seed=6)
    assert out["synthetic_levinson_residual"] < 1e-10
    assert out["synthetic_yw_ar1_err"] < 1e-8
