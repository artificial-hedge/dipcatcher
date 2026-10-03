"""l^p / l^q duality: dual norm computation + Holder bound (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def lp_norm(x: np.ndarray, p: float) -> float:
    return float(np.sum(np.abs(x) ** p) ** (1.0 / p))


def dual_norm(y: np.ndarray, p: float) -> float:
    """Norm of y as functional on l^p: equals l^q norm, 1/p+1/q=1."""
    if p == 1.0:
        return float(np.max(np.abs(y)))
    q = p / (p - 1.0)
    return lp_norm(y, q)


def max_pairing(y: np.ndarray, p: float) -> float:
    """max_{||x||_p=1} y.x computed by unit ball sampling."""
    rng = np.random.default_rng(0)
    best = 0.0
    for _ in range(20000):
        x = rng.standard_normal(y.shape)
        x = x / lp_norm(x, p)
        best = max(best, float(y @ x))
    return best


def _bench_lp_duality(seed: int = 0) -> float:
    checks = []
    y = np.array([3.0, 4.0, 0.0])
    checks.append(abs(dual_norm(y, 2.0) - 5.0) < 1e-9)
    checks.append(abs(dual_norm(y, 1.0) - 4.0) < 1e-9)
    # Holder: |x.y| <= ||x||_p ||y||_q
    x = np.array([1.0, 2.0, -1.0])
    checks.append(abs(x @ y) <= lp_norm(x, 2.0) * dual_norm(y, 2.0) + 1e-9)
    # sampled max pairing approximates dual norm (within 10%)
    mp = max_pairing(y, 2.0)
    checks.append(abs(mp - 5.0) / 5.0 < 0.1)
    # q for p=3 is 1.5: dual_norm(y,3) = (27+64)^(2/3)^(1/1.5)... just compare to direct l^1.5
    checks.append(abs(dual_norm(y, 3.0) - lp_norm(y, 1.5)) < 1e-9)
    return float(sum(checks) / len(checks))


def bench_lp_duality(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lp_duality": _bench_lp_duality(seed)}
