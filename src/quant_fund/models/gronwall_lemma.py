"""Gronwall inequality: u' <= a*u => u(t) <= u0 e^{at} (SYNTHETIC)."""

from __future__ import annotations

import math


def gronwall_bound(u0: float, a: float, t: float) -> float:
    return u0 * math.exp(a * t)


def integral_gronwall(u0: float, a: float, b: float, t: float) -> float:
    """u(t) <= u0 + int(a u + b) => u(t) <= (u0 + b/a)(e^{at}) - b/a."""
    return (u0 + b / a) * math.exp(a * t) - b / a


def _bench_gronwall_lemma(seed: int = 0) -> float:
    checks = []
    # u' = u (equality case): u=e^t equals bound
    checks.append(abs(gronwall_bound(1.0, 1.0, 1.0) - math.e) < 1e-9)
    # u' = 0.5u satisfies u' <= u: e^{0.5} < e
    checks.append(math.exp(0.5) < gronwall_bound(1.0, 1.0, 1.0))
    # integral form: u' = u + 1 => u = 2e^t - 1 equals bound
    checks.append(abs(integral_gronwall(1.0, 1.0, 1.0, 1.0) - (2.0 * math.e - 1.0)) < 1e-9)
    # sub-solution stays under bound: u' = 0.8u + 0 < u + 0.5 pathwise bound
    checks.append(math.exp(0.8) < integral_gronwall(1.0, 1.0, 0.5, 1.0))
    # bound grows monotonic in t
    checks.append(gronwall_bound(1.0, 1.0, 2.0) > gronwall_bound(1.0, 1.0, 1.0))
    # a=0 case -> u(t) <= u0 + bt
    checks.append(abs(integral_gronwall(1.0, 1e-12, 2.0, 3.0) - (1.0 + 6.0)) < 0.05)
    return float(sum(checks) / len(checks))


def bench_gronwall_lemma(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gronwall_lemma": _bench_gronwall_lemma(seed)}
