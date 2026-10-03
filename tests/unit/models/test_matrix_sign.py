import numpy as np

from quant_fund.models.matrix_sign import (
    bench_matrix_sign,
    matrix_sign,
    sign_care,
)


def test_diag_sign():
    S = matrix_sign(np.diag([-2.0, 3.0, -0.5]))
    assert np.abs(S - np.diag([-1.0, 1.0, -1.0])).max() < 1e-10


def test_idempotent():
    rng = np.random.default_rng(0)
    V = rng.standard_normal((4, 4)) + 3 * np.eye(4)
    Z = V @ np.diag([-1.0, 2.0, -3.0, 0.5]) @ np.linalg.inv(V)
    S = matrix_sign(Z)
    S2 = matrix_sign(S)
    assert np.linalg.norm(S2 - S, "fro") < 1e-8


def test_sign_care_matches():
    from quant_fund.models.riccati_care import care_hamiltonian

    A = np.array([[0.0, 1.0], [-2.0, 0.5]])
    B = np.array([[0.0], [1.0]])
    Q = np.eye(2)
    R = np.array([[0.5]])
    Xs = sign_care(A, B, Q, R)
    Xh = care_hamiltonian(A, B, Q, R)
    assert np.linalg.norm(Xs - Xh) / np.linalg.norm(Xh) < 1e-8


def test_bench():
    out = bench_matrix_sign(seed=5)
    assert out["synthetic_sign_diag_err"] < 1e-10
    assert out["synthetic_sign_care_gap"] < 1e-8
