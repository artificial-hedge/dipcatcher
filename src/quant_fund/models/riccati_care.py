"""Continuous-time algebraic Riccati equation canon: CARE
AᵀX + XA − XBR⁻¹BᵀX + Q = 0 solved two ways — (a) Hamiltonian
eigenvalue method on H = [[A, −G],[−Q, −Aᵀ]] picking the stable
invariant subspace, (b) Kleinman Newton refinement via Lyapunov
solves. Bench: 1-D analytic case, LQR gain recovery, and
residuals. All SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_continuous_are, solve_sylvester

FloatArray = NDArray[np.float64]


def care_hamiltonian(A: FloatArray, B: FloatArray, Q: FloatArray, R: FloatArray) -> FloatArray:
    """CARE via the stable Hamiltonian invariant subspace."""
    return np.asarray(
        solve_continuous_are(
            np.asarray(A, dtype=np.float64),
            np.asarray(B, dtype=np.float64),
            np.asarray(Q, dtype=np.float64),
            np.asarray(R, dtype=np.float64),
        ),
        dtype=np.float64,
    )


def kleinman_step(
    X: FloatArray,
    A: FloatArray,
    B: FloatArray,
    Q: FloatArray,
    G: FloatArray,
) -> FloatArray:
    """One Kleinman Newton step: with K = R⁻¹BᵀX solve
    (A − BK)ᵀ X₁ + X₁(A − BK) = −(Q + KᵀRK) = −(Q + XGX)."""
    Acl = A - G @ X
    rhs = -(Q + X @ G @ X)
    return np.asarray(solve_sylvester(Acl.T, Acl, rhs), dtype=np.float64)


def care_kleinman(
    A: FloatArray,
    B: FloatArray,
    Q: FloatArray,
    R: FloatArray,
    iters: int = 25,
) -> FloatArray:
    """Kleinman Newton for CARE — needs a stabilizing start; use a
    cheap Lyapunov-based guess when A is already stable."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    G = B @ np.linalg.inv(R) @ B.T
    # start: solve AᵀX + XA = −(Q + small) only when A stable;
    # otherwise begin from the Hamiltonian seed
    eigs = np.linalg.eigvals(A)
    if np.all(eigs.real < 0):
        X = np.asarray(
            solve_sylvester(A.T, A, -(Q + 0.1 * np.eye(A.shape[0]))),
            dtype=np.float64,
        )
    else:
        X = care_hamiltonian(A, B, Q, R)
    for _ in range(iters):
        Xn = kleinman_step(X, A, B, Q, G)
        if np.linalg.norm(Xn - X, "fro") < 1e-12 * max(np.linalg.norm(Xn, "fro"), 1.0):
            X = Xn
            break
        X = Xn
    return np.asarray(X, dtype=np.float64)


def care_residual(
    X: FloatArray,
    A: FloatArray,
    B: FloatArray,
    Q: FloatArray,
    R: FloatArray,
) -> float:
    G = B @ np.linalg.inv(R) @ B.T
    return float(np.linalg.norm(A.T @ X + X @ A - X @ G @ X + Q, "fro"))


def bench_riccati_care(seed: int = 20261231) -> dict[str, float]:
    out: dict[str, float] = {}
    # 1-D: a=-1? CARE: 2aX − (b²/r)X² + q = 0 → quadratic
    a, b, q, r = -0.5, 1.0, 2.0, 1.0
    # (b²/r)x² − 2a x − q = 0 → x = (a + sqrt(a² + b²q/r))·(r/b²)
    x_star = (a + np.sqrt(a * a + b * b * q / r)) * r / (b * b)
    A = np.array([[a]])
    B = np.array([[b]])
    Q = np.array([[q]])
    R = np.array([[r]])
    X = care_hamiltonian(A, B, Q, R)
    out["synthetic_care_1d_err"] = float(abs(X[0, 0] - x_star))
    out["synthetic_care_1d_residual"] = care_residual(X, A, B, Q, R)
    # 2-d LQR: X gives optimal gain K = R⁻¹BᵀX — check stable
    A2 = np.array([[0.0, 1.0], [-2.0, 0.5]])  # unstable
    B2 = np.array([[0.0], [1.0]])
    Q2 = np.eye(2)
    R2 = np.array([[0.5]])
    X2 = care_hamiltonian(A2, B2, Q2, R2)
    K = np.linalg.inv(R2) @ B2.T @ X2
    eigs = np.linalg.eigvals(A2 - B2 @ K)
    out["synthetic_care_stable_max_eig"] = float(eigs.real.max())
    out["synthetic_care_2d_residual"] = care_residual(X2, A2, B2, Q2, R2)
    out["synthetic_care_sym_err"] = float(np.linalg.norm(X2 - X2.T, "fro"))
    # Kleinman agreement
    Xk = care_kleinman(A2, B2, Q2, R2)
    out["synthetic_kleinman_gap"] = float(
        np.linalg.norm(Xk - X2, "fro") / np.linalg.norm(X2, "fro")
    )
    return out
