"""Kepler's equation solvers: elliptic Newton and bisection oracle."""

from __future__ import annotations

import numpy as np


def kepler_E(M: float, e: float, tol: float = 1e-14, itmax: int = 50) -> float:
    """Solve M = E - e*sin(E) for E via Newton iteration (elliptic)."""
    E = M if e < 0.8 else np.pi
    for _ in range(itmax):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        E -= f / fp
        if abs(f) < tol:
            break
    return float(E)


def _kepler_E_bisect(M: float, e: float, tol: float = 1e-12) -> float:
    lo, hi = M - 1.0, M + 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid - e * np.sin(mid) < M:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def bench_kepler_solve(seed: int = 20261231 + 855) -> dict[str, float]:
    """Residual and bisection-oracle agreement over an (M, e) grid."""
    rng = np.random.default_rng(seed)
    checks = 0.0
    total = 0
    for _ in range(200):
        M = float(rng.uniform(0.0, 2.0 * np.pi))
        e = float(rng.uniform(0.0, 0.95))
        E = kepler_E(M, e)
        total += 2
        checks += float(abs(E - e * np.sin(E) - M) < 1e-10)
        checks += float(abs(E - _kepler_E_bisect(M, e)) < 1e-9)
    return {"synthetic_kepler": checks / total}
