import numpy as np

from quant_fund.models.arnoldi_gmres import arnoldi, bench_arnoldi_gmres, gmres


def test_arnoldi_relation():
    rng = np.random.default_rng(0)
    n = 50
    a = rng.standard_normal((n, n)) + 5 * np.eye(n)
    v, h = arnoldi(lambda x: a @ x, rng.standard_normal(n), 15)
    assert np.allclose(a @ v[:, :15], v[:, :16] @ h[:16, :15], atol=1e-10)


def test_gmres_solves():
    rng = np.random.default_rng(1)
    n = 60
    a = 0.5 * rng.standard_normal((n, n)) + 8 * np.eye(n)
    x_true = rng.standard_normal(n)
    b = a @ x_true
    x, iters, res = gmres(lambda x: a @ x, b, restart=30, max_iter=90)
    assert res < 1e-8
    assert np.linalg.norm(x - x_true) < 1e-6


def test_bench_converges():
    out = bench_arnoldi_gmres(seed=3)
    assert out["synthetic_gmres_residual"] < 1e-8
    assert out["synthetic_arnoldi_relation"] < 1e-10
