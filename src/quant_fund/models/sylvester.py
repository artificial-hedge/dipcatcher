"""Sylvester/Lyapunov canon: Bartels–Stewart algorithm for
AX + XB = C — real Schur decompositions of A and Bᵀ reduce to
quasi-triangular back-substitution; the continuous Lyapunov
equation AᵀP + PA = −Q falls out as the symmetric case. Bench:
residual norms, known-solution recovery, Lyapunov energy
identity for a stable oscillator. All SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_sylvester

FloatArray = NDArray[np.float64]


def sylvester_solve(A: FloatArray, B: FloatArray, C: FloatArray) -> FloatArray:
    """Solve AX + XB = C (thin wrapper over Bartels–Stewart).

    Raises when the equation is singular (λ_i(A) + λ_j(B) = 0).
    """
    X = solve_sylvester(
        np.asarray(A, dtype=np.float64),
        np.asarray(B, dtype=np.float64),
        np.asarray(C, dtype=np.float64),
    )
    return np.asarray(X, dtype=np.float64)


def sylvester_kron(A: FloatArray, B: FloatArray, C: FloatArray) -> FloatArray:
    """Kronecker-form dense solve (I⊗A + Bᵀ⊗I) vec(X) = vec(C) —
    O(n⁶) but unconditionally transparent; used to cross-check
    the Schur path."""
    n, m = A.shape[0], B.shape[0]
    K = np.kron(np.eye(m), A) + np.kron(B.T, np.eye(n))
    x = np.linalg.solve(K, C.reshape(-1, order="F"))
    return np.asarray(x.reshape((n, m), order="F"), dtype=np.float64)


def lyapunov_solve(A: FloatArray, Q: FloatArray) -> FloatArray:
    """Solve AᵀP + PA = −Q for stable A via Bartels–Stewart."""
    return np.asarray(
        sylvester_solve(A.T, np.asarray(A, dtype=np.float64), -np.asarray(Q)),
        dtype=np.float64,
    )


def bench_sylvester(seed: int = 20261231) -> dict[str, float]:
    out: dict[str, float] = {}
    rng = np.random.default_rng(seed)
    # stable A, arbitrary B-shifted: C = AX* + X*B for known X*
    A = rng.standard_normal((5, 5)) - 4 * np.eye(5)
    B = rng.standard_normal((4, 4)) - 4 * np.eye(4)
    X_star = rng.standard_normal((5, 4))
    C = A @ X_star + X_star @ B
    X = sylvester_solve(A, B, C)
    out["synthetic_sylvester_sol_err"] = float(
        np.linalg.norm(X - X_star, "fro") / np.linalg.norm(X_star, "fro")
    )
    out["synthetic_sylvester_residual"] = float(
        np.linalg.norm(A @ X + X @ B - C, "fro") / np.linalg.norm(C, "fro")
    )
    # Kron cross-check
    Xk = sylvester_kron(A, B, C)
    out["synthetic_kron_gap"] = float(np.linalg.norm(X - Xk, "fro") / np.linalg.norm(Xk, "fro"))
    # Lyapunov: AᵀP + PA = −Q for a 2-d damped oscillator —
    # energy decays at rate tr(Q)/(2·tr(P)) roughly; residual check
    A2 = np.array([[-0.5, 1.5], [-1.5, -0.5]])
    Q2 = np.eye(2)
    P = lyapunov_solve(A2, Q2)
    out["synthetic_lyap_residual"] = float(np.linalg.norm(A2.T @ P + P @ A2 + Q2, "fro"))
    # analytic: for this A (normal-ish), P ≈ (1/1.0)·I since
    # A + Aᵀ = −I → P = I solves: check
    out["synthetic_lyap_sol_err"] = float(np.linalg.norm(P - np.eye(2), "fro"))
    return out
