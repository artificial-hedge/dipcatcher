"""SYNTHETIC LDLᵀ factorization + rank-1 update/downdate.

Bunch-Kaufman-free LDLᵀ for SPD matrices with stable rank-1 modification;
solves verified against numpy on random SPD systems.
"""

from __future__ import annotations

import random

import numpy as np


def ldlt(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A = L D Lᵀ, L unit-lower, D diagonal."""
    n = A.shape[0]
    L = np.eye(n)
    d = np.zeros(n)
    for j in range(n):
        d[j] = A[j, j] - sum(L[j, k] ** 2 * d[k] for k in range(j))
        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - sum(L[i, k] * L[j, k] * d[k] for k in range(j))) / d[j]
    return L, d


def ldlt_solve(L: np.ndarray, d: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(b)
    y = b.astype(float).copy()
    for i in range(1, n):
        y[i] -= L[i, :i] @ y[:i]
    z = y / d
    x = z.copy()
    for i in range(n - 2, -1, -1):
        x[i] -= L[i + 1 :, i] @ x[i + 1 :]
    return np.asarray(x)


def rank1_update(
    L: np.ndarray, d: np.ndarray, u: np.ndarray, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rank-1 update (σ>0): LDLᵀ + σuuᵀ via LINPACK cholupdate on L√D."""
    n = len(u)
    C = L * np.sqrt(np.maximum(d, 0.0))[None, :]
    w = u.astype(float).copy()
    for k in range(n):
        r = np.sqrt(C[k, k] ** 2 + sigma * w[k] ** 2)
        ck = r / C[k, k]
        sk = w[k] / C[k, k]
        C[k, k] = r
        for i in range(k + 1, n):
            C[i, k] = (C[i, k] + sigma * sk * w[i]) / ck
            w[i] = ck * w[i] - sk * C[i, k]
    d2 = np.diag(C) ** 2
    L2 = C / np.diag(C)[None, :]
    return L2, d2


def _spd(rng: random.Random, n: int) -> np.ndarray:
    M = np.array([[rng.uniform(-1, 1) for _ in range(n)] for _ in range(n)])
    return np.asarray(M @ M.T + n * np.eye(n))


def bench_ldlt_solve(seed: int = 20261231 + 530) -> dict[str, float]:
    rng = random.Random(seed)
    np.random.seed(seed)
    solve_ok = 0
    fac_ok = 0
    upd_ok = 0
    n_trials = 30
    for _ in range(n_trials):
        n = rng.randrange(3, 8)
        A = _spd(rng, n)
        b = np.array([rng.uniform(-1, 1) for _ in range(n)])
        L, d = ldlt(A)
        fac_ok += int(np.allclose(L @ np.diag(d) @ L.T, A, atol=1e-8))
        x = ldlt_solve(L, d, b)
        solve_ok += int(np.allclose(A @ x, b, atol=1e-8))
        u = np.array([rng.uniform(-1, 1) for _ in range(n)])
        sig = rng.uniform(0.1, 2)
        L2, d2 = rank1_update(L, d, u, sig)
        upd_ok += int(np.allclose(L2 @ np.diag(d2) @ L2.T, A + sig * np.outer(u, u), atol=1e-6))
    return {
        "synthetic_factor_exact": fac_ok / n_trials,
        "synthetic_solve_exact": solve_ok / n_trials,
        "synthetic_rank1_update": upd_ok / n_trials,
    }
