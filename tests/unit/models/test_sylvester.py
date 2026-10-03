import numpy as np

from quant_fund.models.sylvester import (
    bench_sylvester,
    lyapunov_solve,
    sylvester_kron,
    sylvester_solve,
)


def test_solve_residual():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 4)) - 3 * np.eye(4)
    B = rng.standard_normal((3, 3)) - 3 * np.eye(3)
    C = rng.standard_normal((4, 3))
    X = sylvester_solve(A, B, C)
    assert np.linalg.norm(A @ X + X @ B - C) < 1e-10


def test_kron_agrees():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((3, 3)) - 3 * np.eye(3)
    B = rng.standard_normal((3, 3)) - 3 * np.eye(3)
    C = rng.standard_normal((3, 3))
    assert np.linalg.norm(sylvester_solve(A, B, C) - sylvester_kron(A, B, C)) < 1e-10


def test_lyapunov_stable():
    A = np.array([[-0.5, 1.5], [-1.5, -0.5]])
    P = lyapunov_solve(A, np.eye(2))
    assert np.linalg.norm(A.T @ P + P @ A + np.eye(2)) < 1e-12
    assert np.linalg.norm(P - np.eye(2)) < 1e-10


def test_bench():
    out = bench_sylvester(seed=3)
    assert out["synthetic_sylvester_residual"] < 1e-10
    assert out["synthetic_lyap_sol_err"] < 1e-10
