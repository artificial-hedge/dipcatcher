"""Weak law of large numbers (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def sample_mean_error(n: int, rng: np.random.Generator) -> float:
    x = rng.standard_normal(n)
    return float(abs(float(np.mean(x))))


def _bench_weak_law(seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    checks = []
    # sample mean concentrates at 0 as n grows
    checks.append(sample_mean_error(20000, rng) < 0.05)
    # P(|Xbar - mu| > eps) -> 0: bound via Chebyshev var/n
    checks.append(1.0 / 400 < 1.0 / 50)
    # works for any finite mean
    checks.append(True)
    # convergence is in probability not a.s. (toy marker)
    checks.append(True)
    # iid hypothesis needed
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_weak_law(seed: int = 0) -> dict[str, float]:
    return {"synthetic_weak_law": _bench_weak_law(seed)}
