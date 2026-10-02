import numpy as np

from quant_fund.models.riccati_care import (
    bench_riccati_care,
    care_hamiltonian,
    care_kleinman,
)


def test_1d_analytic():
    A = np.array([[-0.5]])
    B = np.array([[1.0]])
    Q = np.array([[2.0]])
    R = np.array([[1.0]])
    X = care_hamiltonian(A, B, Q, R)
    x_star = -0.5 + np.sqrt(0.25 + 2.0)
    assert abs(X[0, 0] - x_star) < 1e-10


def test_stabilizing_gain():
    A = np.array([[0.0, 1.0], [-2.0, 0.5]])
    B = np.array([[0.0], [1.0]])
    X = care_hamiltonian(A, B, np.eye(2), np.array([[0.5]]))
    K = 2.0 * B.T @ X
    assert np.linalg.eigvals(A - B @ K).real.max() < 0


def test_kleinman_agrees():
    A = np.array([[-1.0, 1.0], [0.0, -2.0]])
    B = np.array([[0.0], [1.0]])
    Xh = care_hamiltonian(A, B, np.eye(2), np.eye(1))
    Xk = care_kleinman(A, B, np.eye(2), np.eye(1))
    assert np.linalg.norm(Xh - Xk) / np.linalg.norm(Xh) < 1e-8


def test_bench():
    out = bench_riccati_care(seed=4)
    assert out["synthetic_care_1d_err"] < 1e-10
    assert out["synthetic_care_2d_residual"] < 1e-10
    assert out["synthetic_care_stable_max_eig"] < 0
