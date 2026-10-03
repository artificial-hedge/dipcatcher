"""Matrix-square-root canon: Denman–Beavers coupled iteration for
the principal square root of an SPD/no-nonpositive-eigenvalue
matrix — X_{k+1} = (X_k + Y_k^{-1})/2, Y_{k+1} = (Y_k +
X_k^{-1})/2 starting from X_0 = A, Y_0 = I; converges
quadratically. Bench: residual ||S²−A|| on random SPD systems
plus a diagonal-dominant stress case. All SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def matrix_sqrt(A: FloatArray, iters: int = 40, tol: float = 1e-12) -> FloatArray:
    """Principal matrix square root via Denman–Beavers."""
    A = np.asarray(A, dtype=np.float64)
    X = A.copy()
    Y = np.eye(A.shape[0])
    for _ in range(iters):
        Xi = np.linalg.inv(X)
        Yi = np.linalg.inv(Y)
        X_new = 0.5 * (X + Yi)
        Y_new = 0.5 * (Y + Xi)
        if np.linalg.norm(X_new - X, "fro") < tol * np.linalg.norm(X_new, "fro"):
            X = X_new
            break
        X, Y = X_new, Y_new
    return np.asarray(X, dtype=np.float64)


def sqrtm_newton(A: FloatArray, iters: int = 60) -> FloatArray:
    """Newton iteration for X² = A: X_{k+1} = (X_k + X_k^{-1}A)/2
    from X_0 = I — equivalent to the Denman–Beavers Y-chain."""
    A = np.asarray(A, dtype=np.float64)
    X = np.eye(A.shape[0])
    for _ in range(iters):
        X = 0.5 * (X + np.linalg.inv(X) @ A)
        if np.linalg.norm(X @ X - A, "fro") < 1e-12:
            break
    return np.asarray(X, dtype=np.float64)


def bench_matrix_sqrt(seed: int = 20261231) -> dict[str, float]:
    out: dict[str, float] = {}
    rng = np.random.default_rng(seed)
    # random SPD
    B = rng.standard_normal((6, 6))
    A = B @ B.T + 0.5 * np.eye(6)
    S = matrix_sqrt(A)
    out["synthetic_sqrt_residual"] = float(
        np.linalg.norm(S @ S - A, "fro") / np.linalg.norm(A, "fro")
    )
    # symmetry of the principal root
    out["synthetic_sqrt_sym_err"] = float(np.linalg.norm(S - S.T, "fro"))
    # commutation: S·A = A·S for the principal root
    out["synthetic_sqrt_comm_err"] = float(np.linalg.norm(S @ A - A @ S, "fro"))
    # diagonal: exact check
    D = np.diag([4.0, 9.0, 0.25])
    Sd = matrix_sqrt(D)
    out["synthetic_sqrt_diag_err"] = float(np.abs(Sd - np.diag([2.0, 3.0, 0.5])).max())
    # Newton-form cross-check
    Sp = sqrtm_newton(A)
    out["synthetic_sqrt_prod_err"] = float(
        np.linalg.norm(Sp @ Sp - A, "fro") / np.linalg.norm(A, "fro")
    )
    return out
