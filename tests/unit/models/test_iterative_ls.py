import numpy as np

from quant_fund.models.iterative_ls import (
    bench_iterative_ls,
    cgls,
    lsqr,
)


def _problem(seed=0):
    rng = np.random.default_rng(seed)
    n, d = 200, 30
    a = rng.standard_normal((n, d))
    x = rng.standard_normal(d)
    b = a @ x
    return a, x, b


def test_cgls_solves():
    a, x, b = _problem()
    xc, _ = cgls(lambda v: a @ v, lambda v: a.T @ v, b, it=300, tol=1e-12)
    assert np.linalg.norm(a @ xc - b) / np.linalg.norm(b) < 1e-6


def test_lsqr_solves():
    a, x, b = _problem()
    xl, _ = lsqr(lambda v: a @ v, lambda v: a.T @ v, b, it=300, tol=1e-12)
    assert np.linalg.norm(a @ xl - b) / np.linalg.norm(b) < 1e-6


def test_lsqr_overdetermined_noise():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((300, 20))
    b = a @ rng.standard_normal(20) + 0.05 * rng.standard_normal(300)
    xd = np.linalg.lstsq(a, b, rcond=None)[0]
    xl, _ = lsqr(lambda v: a @ v, lambda v: a.T @ v, b, it=500)
    assert np.linalg.norm(xl - xd) < 0.05


def test_bench_keys():
    out = bench_iterative_ls()
    assert out["synthetic_lsqr_err"] < 0.01
