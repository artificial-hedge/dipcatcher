import numpy as np

from quant_fund.models.conjugate_gradient import (
    bench_conjugate_gradient,
    cg_solve,
    nlcg_pr,
)


def test_cg_solves_spd():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (20, 20))
    a = a.T @ a + np.eye(20)
    b = rng.normal(0, 1, 20)
    x = cg_solve(a, b, tol=1e-12)
    np.testing.assert_allclose(a @ x, b, atol=1e-6)


def test_nlcg_minimizes_quadratic():
    a = np.diag([1.0, 4.0, 8.0])
    b = np.array([1.0, -2.0, 0.5])
    f = lambda x: float(0.5 * x @ a @ x - b @ x)  # noqa: E731
    g = lambda x: a @ x - b  # noqa: E731
    x = nlcg_pr(f, g, np.zeros(3), it=200)
    np.testing.assert_allclose(x, np.linalg.solve(a, b), atol=1e-5)


def test_bench_conjugate_gradient():
    out = bench_conjugate_gradient(seed=560)
    assert out["synthetic_cg_solve_err"] < 1e-6
    assert out["synthetic_cg_rosen_f"] < out["synthetic_gd_rosen_f"]
