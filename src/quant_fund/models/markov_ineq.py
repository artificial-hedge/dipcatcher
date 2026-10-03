"""Markov + Chebyshev + Chernoff inequalities verified on samples (SYNTHETIC)."""

from __future__ import annotations

import math


def markov_bound(mean: float, a: float) -> float:
    return mean / a


def chebyshev_bound(var: float, a: float) -> float:
    return var / (a * a)


def chernoff_uniform(p: float, n: int, delta: float) -> float:
    """Hoeffding for mean of uniform bounded in [0,1]."""
    return math.exp(-2 * n * delta * delta)


def empirical_tail(samples: list[float], a: float) -> float:
    return sum(1 for s in samples if s >= a) / len(samples)


def _bench_markov_ineq(seed: int = 0) -> float:
    checks = []
    import numpy as np

    rng = np.random.default_rng(seed)
    x = rng.exponential(1.0, 200_000)
    mean = float(np.mean(x))
    # Markov: P(X >= 3) <= mean/3 ~ 0.33; actual ~ e^-3 ~ 0.05
    tail = float(np.mean(x >= 3.0))
    checks.append(tail <= markov_bound(mean, 3.0) + 1e-9)
    # Chebyshev: P(|X-mu| >= 2sigma) <= 1/4
    z = rng.standard_normal(200_000)
    checks.append(float(np.mean(np.abs(z) >= 2.0)) <= 0.25)
    # empirical tail of exponential ~ e^-3
    checks.append(abs(tail - math.exp(-3)) < 0.01)
    # Hoeffding bound on uniform mean deviation
    checks.append(chernoff_uniform(0.5, 100, 0.1) < 0.14)
    # Chernoff tighter than Chebyshev at large delta
    checks.append(chernoff_uniform(0.5, 100, 0.3) < chebyshev_bound(1.0 / 12 / 100, 0.3))
    checks.append(markov_bound(1.0, 2.0) == 0.5)
    return float(sum(checks) / len(checks))


def bench_markov_ineq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_markov_ineq": _bench_markov_ineq(seed)}
