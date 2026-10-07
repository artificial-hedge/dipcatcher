"""Randomized QB factorization: A ≈ Q B for low-rank approximation (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 663


def rand_qb(A: np.ndarray, rank: int, rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    n = A.shape[1]
    Omega = rng.rand(n, rank)
    Y = A @ Omega
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ A
    return Q, B


def bench_randomized_qb(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 20
    for _ in range(trials):
        n = 30
        U = np.linalg.qr(rng.rand(n, n))[0]
        sv = np.exp(-np.arange(n) * 0.4)  # fast-decaying spectrum
        A = U @ np.diag(sv) @ U.T
        Q, B = rand_qb(A, 14, rng)
        err = np.linalg.norm(A - Q @ B) / np.linalg.norm(A)
        ok += float(err < 0.08)
    return {"synthetic_qb_lowrank": ok / trials}
