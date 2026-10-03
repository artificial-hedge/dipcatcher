"""Cyclic coordinate descent for smooth convex objectives."""

import numpy as np

_SEED = 20261231 + 574


def coord_descent_qp(Q: np.ndarray, c: np.ndarray, iters: int = 300) -> np.ndarray:
    """min ½xᵀQx + cᵀx via exact coordinate minimization."""
    n = len(c)
    x = np.zeros(n)
    for _ in range(iters):
        for j in range(n):
            x[j] = -(Q[j] @ x - Q[j, j] * x[j] + c[j]) / Q[j, j]
    return x


def bench_coord_descent(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    n = 50
    ok = 0
    for _ in range(n):
        m_ = 4
        M = rng.uniform(-1, 1, (m_, m_))
        Q = M @ M.T + np.eye(m_)
        c = rng.uniform(-2, 2, m_)
        x = coord_descent_qp(Q, c)
        ref = -np.linalg.solve(Q, c)
        ok += int(np.allclose(x, ref, atol=1e-4))
    return {"synthetic_qp_exact": float(ok / n)}
