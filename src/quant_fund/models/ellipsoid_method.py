"""Ellipsoid method for feasibility of Ax ≤ b (as optimization proxy)."""

import numpy as np

_SEED = 20261231 + 571


def ellipsoid_feasible(A: np.ndarray, b: np.ndarray, iters: int = 4000) -> np.ndarray | None:
    n = A.shape[1]
    x = np.zeros(n)
    P = np.eye(n) * 100.0
    for _ in range(iters):
        v = A @ x - b
        i = int(np.argmax(v))
        if v[i] <= 1e-7:
            return x
        g = A[i] / np.sqrt(A[i] @ P @ A[i])
        Pg = P @ g
        x = x - Pg / (n + 1)
        P = (n * n / (n * n - 1)) * (P - (2.0 / (n + 1)) * np.outer(Pg, Pg))
        w, V = np.linalg.eigh(P)
        w = np.maximum(w, 1e-12)
        P = (V * w) @ V.T
    return None


def bench_ellipsoid_method(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    n = 40
    ok = 0
    for _ in range(n):
        nv, nc = 4, 6
        A = rng.uniform(-2, 2, (nc, nv))
        x0 = rng.uniform(-1, 2, nv)
        b = A @ x0 + rng.uniform(0.05, 1, nc)
        x = ellipsoid_feasible(A, b)
        ok += int(x is not None and np.all(A @ x <= b + 1e-4))
    return {"synthetic_feasible_point": float(ok / n)}
