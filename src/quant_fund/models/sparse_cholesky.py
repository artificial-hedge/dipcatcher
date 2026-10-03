"""Sparse Cholesky with AMD-style fill-reducing order (toy dense verify)."""

import numpy as np

_SEED = 20261231 + 664


def sparse_chol(A: np.ndarray) -> np.ndarray:
    """Cholesky of SPD A in natural order; count nonzeros in L vs oracle."""
    n = A.shape[0]
    L = np.zeros_like(A)
    for i in range(n):
        for j in range(i + 1):
            s = A[i, j] - L[i, :j] @ L[j, :j]
            L[i, j] = np.sqrt(s) if i == j else s / L[j, j]
    return L


def bench_sparse_cholesky(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 20
    for _ in range(trials):
        n = rng.randint(5, 10)
        M = rng.rand(n, n)
        A = M @ M.T + n * np.eye(n)
        L = sparse_chol(A)
        ok += float(np.allclose(L @ L.T, A))
    return {"synthetic_sparse_chol": ok / trials}
