"""Projected gradient descent for box/simplex-constrained convex objectives."""

import numpy as np

_SEED = 20261231 + 575


def _proj_simplex(v: np.ndarray) -> np.ndarray:
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    theta = (css[rho] - 1) / (rho + 1.0)
    return np.asarray(np.maximum(v - theta, 0))


def proj_grad_quad(Q: np.ndarray, c: np.ndarray, iters: int = 600) -> np.ndarray:
    """min ½xᵀQx + cᵀx on the probability simplex."""
    x = np.ones(len(c)) / len(c)
    L = float(np.linalg.eigvalsh(Q).max())
    for _ in range(iters):
        x = _proj_simplex(x - (Q @ x + c) / L)
    return x


def _cvx_oracle(Q: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Solve via KKT scan over support subsets (n small)."""
    import itertools

    n = len(c)
    best, bx = np.inf, None
    for k in range(1, n + 1):
        for S in itertools.combinations(range(n), k):
            idx = list(S)
            Qs = Q[np.ix_(idx, idx)]
            cs = c[idx]
            # KKT: Q_S x + c + ν1 = 0, 1ᵀx=1 → solve 2x2 block
            A_ = np.block([[Qs, np.ones((k, 1))], [np.ones((1, k)), np.zeros((1, 1))]])
            try:
                sol = np.linalg.solve(A_, np.concatenate([-cs, [1.0]]))
            except np.linalg.LinAlgError:
                continue
            xk = sol[:k]
            if np.any(xk < -1e-6):
                continue
            x = np.zeros(n)
            x[idx] = np.maximum(xk, 0)
            if abs(x.sum() - 1) > 1e-6:
                continue
            obj = 0.5 * x @ Q @ x + c @ x
            if obj < best:
                best, bx = obj, x
    return bx if bx is not None else np.ones(n) / n


def bench_proj_gradient(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    n = 40
    ok = 0
    for _ in range(n):
        m_ = 4
        M = rng.uniform(-1, 1, (m_, m_))
        Q = M @ M.T + np.eye(m_) * 0.5
        c = rng.uniform(-1, 1, m_)
        x = proj_grad_quad(Q, c)
        ref = _cvx_oracle(Q, c)
        obj = 0.5 * x @ Q @ x + c @ x
        obj_ref = 0.5 * ref @ Q @ ref + c @ ref
        ok += int(np.isclose(obj, obj_ref, atol=1e-4) and abs(x.sum() - 1) < 1e-6)
    return {"synthetic_simplex_optimal": float(ok / n)}
