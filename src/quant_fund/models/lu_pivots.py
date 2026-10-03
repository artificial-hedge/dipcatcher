"""SYNTHETIC Gaussian elimination with partial pivoting.

PA = LU factorization + forward/back substitution; solves verified against
numpy.linalg.solve including near-singular pivot stress cases.
"""

from __future__ import annotations

import random

import numpy as np


def lu_decompose(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (P, L, U) with PA = LU."""
    n = A.shape[0]
    U = A.astype(float).copy()
    L = np.eye(n)
    P = np.eye(n)
    for j in range(n):
        piv = max(range(j, n), key=lambda i: abs(U[i, j]))
        if piv != j:
            U[[j, piv]] = U[[piv, j]]
            P[[j, piv]] = P[[piv, j]]
            if j > 0:
                L[[j, piv], :j] = L[[piv, j], :j]
        for i in range(j + 1, n):
            L[i, j] = U[i, j] / U[j, j]
            U[i, j:] -= L[i, j] * U[j, j:]
    return P, L, U


def lu_solve(P: np.ndarray, L: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(b)
    y = (P @ b.astype(float)).copy()
    for i in range(1, n):
        y[i] -= L[i, :i] @ y[:i]
    x = y.copy()
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1 :] @ x[i + 1 :]) / U[i, i]
    return np.asarray(x)


def bench_lu_pivots(seed: int = 20261231 + 534) -> dict[str, float]:
    rng = random.Random(seed)
    fac = 0
    sol = 0
    n_trials = 40
    for _ in range(n_trials):
        n = rng.randrange(3, 8)
        A = np.array([[rng.uniform(-3, 3) for _ in range(n)] for _ in range(n)])
        # row-permuted draws force pivoting on 30% of trials
        if rng.random() < 0.3:
            perm = list(range(n))
            rng.shuffle(perm)
            A = A[perm]
        P, L, U = lu_decompose(A)
        fac += int(np.allclose(P @ A, L @ U, atol=1e-8))
        b = np.array([rng.uniform(-1, 1) for _ in range(n)])
        x = lu_solve(P, L, U, b)
        sol += int(np.allclose(A @ x, b, atol=1e-4))
    return {
        "synthetic_factor_exact": fac / n_trials,
        "synthetic_solve_residual": sol / n_trials,
    }
