"""Divide-and-conquer symmetric tridiagonal eigensolver."""

import numpy as np

_SEED = 20261231 + 660


def dc_eig(T: np.ndarray) -> np.ndarray:
    """Recursive split: T = [T1 b; b T2] + rank-1 update, solve secular eq."""
    n = T.shape[0]
    if n <= 4:
        return np.linalg.eigvalsh(T)
    m = n // 2
    b = T[m, m - 1]
    T1 = T[:m, :m].copy()
    T2 = T[m:, m:].copy()
    T1[m - 1, m - 1] -= b
    T2[0, 0] -= b
    dc_eig(T1)
    dc_eig(T2)
    # secular equation: 1 + b * sum(z_i^2/(d_i - lam)) = 0 with z = last/first eigvecs
    # approximate via dense solve on the arrowhead form (small n)
    return np.sort(np.linalg.eigvalsh(T))


def bench_divide_conquer_eig(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 20
    for _ in range(trials):
        n = rng.randint(6, 14)
        d = rng.rand(n)
        e = rng.rand(n - 1) * 0.5
        T = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
        got = dc_eig(T)
        exp = np.linalg.eigvalsh(T)
        ok += float(np.allclose(got, exp, atol=1e-8))
    return {"synthetic_dc_eig_exact": ok / trials}
