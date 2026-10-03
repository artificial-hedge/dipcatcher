"""Primal log-barrier interior point for min cᵀx s.t. Ax = b, x ≥ 0."""

import numpy as np

_SEED = 20261231 + 572


def barrier_lp(c: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Newton on min cᵀx - t·Σ log x s.t. Ax = b, increasing t."""
    n = len(c)
    x = np.ones(n)
    # project to feasibility
    x = x + A.T @ np.linalg.solve(A @ A.T + 1e-9 * np.eye(A.shape[0]), b - A @ x)
    x = np.maximum(x, 0.5)
    for _outer in range(60):
        t = 10.0 ** (np.linspace(1.5, -3, 60)[_outer])
        for _ in range(25):
            g = c - t / x
            # KKT: [H Aᵀ; A 0] [dx; dy] = [-g; 0]
            AHinv = A @ np.diag(x**2 / t)
            lam = np.linalg.solve(
                AHinv @ A.T + 1e-10 * np.eye(A.shape[0]), -A @ np.diag(x**2 / t) @ g
            )
            dx = -np.diag(x**2 / t) @ (g + A.T @ lam)
            alpha = 1.0
            while np.any(x + alpha * dx <= 0):
                alpha *= 0.5
                if alpha < 1e-8:
                    break
            x = x + 0.99 * alpha * dx
    # polish: alternate projection onto Ax = b and x ≥ 0 (POCS)
    for _ in range(20):
        x = x + A.T @ np.linalg.solve(A @ A.T + 1e-9 * np.eye(A.shape[0]), b - A @ x)
        x = np.maximum(x, 0.0)
    return np.asarray(x)


def bench_barrier_ip(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    n = 30
    ok = 0
    for _ in range(n):
        n_v = 4
        # random equality-const LP with known feasible point
        A = rng.uniform(-1, 2, (2, n_v))
        x0 = rng.uniform(0.5, 2, n_v)
        b = A @ x0
        c = rng.uniform(0.1, 2, n_v)
        # bounded objective region: add x_i ≤ 5 via c direction... use grid oracle
        x = barrier_lp(c, A, b)
        feas = np.allclose(A @ x, b, atol=1e-3) and np.all(x > -1e-6)
        # oracle: vertex enumeration (n choose 2 active x_i = 0)
        import itertools

        best = np.inf
        for comb in itertools.combinations(range(n_v), n_v - A.shape[0]):
            M = np.vstack([A] + [[1.0 if k == j else 0.0 for k in range(n_v)] for j in comb])
            try:
                xv = np.linalg.solve(M, np.concatenate([b, np.zeros(len(comb))]))
            except np.linalg.LinAlgError:
                continue
            if np.all(xv >= -1e-7) and np.allclose(A @ xv, b, atol=1e-6):
                best = min(best, float(c @ xv))
        if np.isinf(best):
            continue
        ok += int(feas and c @ x <= best + 0.08)
    return {"synthetic_ip_optimal": float(ok / n)}
