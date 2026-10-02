import numpy as np

from quant_fund.models.matrix_sqrt import (
    bench_matrix_sqrt,
    matrix_sqrt,
    sqrtm_newton,
)


def test_spd_residual():
    rng = np.random.default_rng(0)
    B = rng.standard_normal((5, 5))
    A = B @ B.T + np.eye(5)
    S = matrix_sqrt(A)
    assert np.linalg.norm(S @ S - A, "fro") < 1e-10


def test_diag():
    S = matrix_sqrt(np.diag([4.0, 16.0, 0.04]))
    assert np.abs(S - np.diag([2.0, 4.0, 0.2])).max() < 1e-10


def test_newton_form():
    rng = np.random.default_rng(2)
    B = rng.standard_normal((4, 4))
    A = B @ B.T + np.eye(4)
    S = sqrtm_newton(A)
    assert np.linalg.norm(S @ S - A, "fro") < 1e-10


def test_bench():
    out = bench_matrix_sqrt(seed=2)
    assert out["synthetic_sqrt_residual"] < 1e-10
    assert out["synthetic_sqrt_diag_err"] < 1e-10
