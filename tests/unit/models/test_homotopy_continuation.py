import numpy as np

from quant_fund.models.homotopy_continuation import (
    bench_homotopy,
    newton_homotopy,
    newton_solve,
    pc_continuation,
)


def test_newton_solves_easy():
    f = lambda x: x**2 - 2.0  # noqa: E731
    jac = lambda x: np.diag(2.0 * x)  # noqa: E731
    x, ok = newton_solve(f, jac, np.array([1.0]))
    assert ok
    assert abs(x[0] - np.sqrt(2.0)) < 1e-10


def test_homotopy_recovers_hard_root():
    rng = np.random.default_rng(0)
    d = 4
    a = rng.standard_normal((d, d)) + 5.0 * np.eye(d)
    xs = rng.standard_normal(d)

    def f(x):
        return a @ x - a @ xs + 2.0 * np.sin(3.0 * (x - xs))

    def jac(x):
        return a + 6.0 * np.diag(np.cos(3.0 * (x - xs)))

    x0 = xs + 6.0 * rng.standard_normal(d)
    xh, _ = newton_homotopy(f, jac, x0, x0, steps=12)
    assert np.linalg.norm(xh - xs) < 1e-8


def test_pc_continuation():
    rng = np.random.default_rng(1)
    d = 4
    a = rng.standard_normal((d, d)) + 5.0 * np.eye(d)
    xs = rng.standard_normal(d)

    def f(x):
        return a @ x - a @ xs + 2.0 * np.sin(3.0 * (x - xs))

    def jac(x):
        return a + 6.0 * np.diag(np.cos(3.0 * (x - xs)))

    x0 = xs + 6.0 * rng.standard_normal(d)
    xp, _ = pc_continuation(f, jac, x0, steps=30)
    assert np.linalg.norm(xp - xs) < 1e-8


def test_bench_keys():
    out = bench_homotopy()
    assert out["synthetic_homotopy_err"] < 1e-8
    assert out["synthetic_pc_err"] < 1e-8
