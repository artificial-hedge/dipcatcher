import numpy as np

from quant_fund.models.ode_solvers import (
    bench_ode_solvers,
    euler_maruyama,
    implicit_midpoint,
    rk4_solve,
    rk45_solve,
)


def test_rk4_decay():
    y = rk4_solve(lambda t, y: -2 * y, 0.0, 1.0, np.array([1.0]), 200)
    assert abs(y[0] - np.exp(-2)) < 1e-9


def test_rk45_adaptive():
    y, steps = rk45_solve(lambda t, y: -2 * y, 0.0, 1.0, np.array([1.0]), tol=1e-7)
    assert abs(y[0] - np.exp(-2)) < 1e-3
    assert steps < 5000


def test_implicit_midpoint_symplectic_decay():
    y = implicit_midpoint(lambda t, y: -2 * y, 0.0, 1.0, np.array([1.0]), 100)
    assert abs(y[0] - np.exp(-2)) < 1e-4


def test_euler_maruyama_runs():
    rng = np.random.default_rng(0)
    y = euler_maruyama(
        lambda t, y: -y,
        lambda t, y: 0.1 * np.ones_like(y),
        0.0,
        1.0,
        np.array([1.0]),
        100,
        rng,
    )
    assert np.isfinite(y).all()


def test_bench_keys():
    out = bench_ode_solvers()
    assert out["synthetic_rk4_err"] < 1e-8
    assert out["synthetic_rk45_logistic_err"] < 1e-4
    assert out["synthetic_em_gbm_mean_err"] < 0.12
    assert out["synthetic_mil_gbm_mean_err"] < 0.12
