"""Simplex tableau method for max LP: cᵀx s.t. Ax ≤ b, x ≥ 0 (SYNTHETIC)."""

import itertools

import numpy as np

_SEED = 20261231 + 570


def simplex(c: np.ndarray, A: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray]:
    """Bland's-rule simplex on standard form with slack."""
    m, n = A.shape
    T = np.zeros((m + 1, n + m + 1))
    T[:m, :n] = A
    T[:m, n : n + m] = np.eye(m)
    T[:m, -1] = b
    T[-1, :n] = -c
    basis = list(range(n, n + m))
    for _ in range(1000):
        row = T[-1, :-1]
        neg = np.where(row < -1e-10)[0]
        if len(neg) == 0:
            break
        j = neg[0]
        col = T[:m, j]
        pos = np.where(col > 1e-10)[0]
        if len(pos) == 0:
            raise ValueError("unbounded")
        ratios = T[:m, -1][pos] / col[pos]
        i = pos[np.argmin(ratios)]
        piv = T[i, j]
        T[i] /= piv
        for k in range(m + 1):
            if k != i:
                T[k] -= T[k, j] * T[i]
        basis[i] = j
    x = np.zeros(n + m)
    for i, bi in enumerate(basis):
        x[bi] = T[i, -1]
    return float(T[-1, -1]), x[:n]


def _brute(c: np.ndarray, A: np.ndarray, b: np.ndarray) -> float:
    """Enumerate all vertex intersections."""
    m, n = A.shape
    best = -np.inf
    eqs = [row for row in A] + [-np.eye(n)[i] for i in range(n)]
    rhs = list(b) + [0.0] * n
    for comb in itertools.combinations(range(len(eqs)), n):
        M = np.array([eqs[i] for i in comb])
        if abs(np.linalg.det(M)) < 1e-9:
            continue
        x = np.linalg.solve(M, np.array([rhs[i] for i in comb]))
        if np.all(A @ x <= b + 1e-7) and np.all(x >= -1e-9):
            best = max(best, float(c @ x))
    return best


def bench_simplex_lp(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    n = 40
    ok = 0
    for _ in range(n):
        n_v, n_c = 3, 4
        A = rng.uniform(-1, 2, (n_c, n_v))
        x0 = rng.uniform(0.5, 2, n_v)
        b = A @ x0 + rng.uniform(0.1, 2, n_c)
        c = rng.uniform(-2, 3, n_v)
        A = np.vstack([A, np.eye(n_v)])
        b = np.concatenate([b, np.full(n_v, 20.0)])
        try:
            obj, _ = simplex(c, A, b)
            ref = _brute(c, A, b)
            ok += int(np.isclose(obj, ref, atol=1e-6))
        except ValueError:
            ok += 0
    return {"synthetic_optimum_exact": float(ok / n)}
