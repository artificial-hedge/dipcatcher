"""Concentration inequalities: Hoeffding / McDiarmid bounds (SYNTHETIC)."""

from __future__ import annotations

import math


def hoeffding_bound(n: int, eps: float, rng: float = 1.0) -> float:
    """P(|S_n/n - mu| >= eps) <= 2 exp(-2 n eps^2 / rng^2)."""
    return 2.0 * math.exp(-2.0 * n * eps * eps / (rng * rng))


def mcdiarmid_bound(eps: float, c_sq: float) -> float:
    """Bounded-difference: P(|f - E f| >= eps) <= 2 exp(-2 eps^2 / sum c_i^2)."""
    return 2.0 * math.exp(-2.0 * eps * eps / c_sq)


def subgauss_mgf(t: float, sigma: float) -> float:
    """Subgaussian MGF bound: E e^{tX} <= e^{t^2 sigma^2 / 2}."""
    return math.exp(t * t * sigma * sigma / 2.0)


def _bench_concentration_ineq(seed: int = 0) -> float:
    checks = []
    # n=100, eps=0.2, range 1: bound 2 e^{-8} ~ 6.7e-4
    b = hoeffding_bound(100, 0.2)
    checks.append(b < 1e-3)
    # small eps or n gives trivial bound > 1
    checks.append(hoeffding_bound(1, 0.01) > 1.0)
    # McDiarmid with small total sensitivity
    checks.append(mcdiarmid_bound(1.0, 0.1) < 0.001)
    # bound decreasing in n
    checks.append(hoeffding_bound(200, 0.2) < hoeffding_bound(50, 0.2))
    # subgaussian MGF at t=0 is 1, grows with sigma
    checks.append(subgauss_mgf(0.0, 1.0) == 1.0)
    checks.append(subgauss_mgf(1.0, 2.0) > subgauss_mgf(1.0, 1.0))
    return float(sum(checks) / len(checks))


def bench_concentration_ineq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_concentration_ineq": _bench_concentration_ineq(seed)}
