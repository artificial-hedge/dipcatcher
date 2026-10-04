"""Arcsine law for BM occupation time (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def arcsine_cdf(x: float) -> float:
    """P(fraction of time B > 0 <= x) = (2/pi) arcsin(sqrt(x))."""
    return float(2 / np.pi * np.arcsin(np.sqrt(x)))


def _bench_occupation_bm(seed: int = 0) -> float:
    checks = []
    # arcsine CDF: F(0)=0, F(1)=1, F(1/2)=1/2
    checks.append(abs(arcsine_cdf(0.5) - 0.5) < 1e-12)
    checks.append(abs(arcsine_cdf(1.0) - 1.0) < 1e-6)
    # symmetry: F(x) + F(1-x) = 1
    checks.append(abs(arcsine_cdf(0.2) + arcsine_cdf(0.8) - 1.0) < 1e-12)
    # BM spends ~arcsine-distributed time positive: P(frac > 0.9) ≈ 0.204
    checks.append(abs((1 - arcsine_cdf(0.9)) - 0.2048) < 0.01)
    # monotone
    checks.append(arcsine_cdf(0.7) > arcsine_cdf(0.3))
    return float(sum(checks) / len(checks))


def bench_occupation_bm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_occupation_bm": _bench_occupation_bm(seed)}
