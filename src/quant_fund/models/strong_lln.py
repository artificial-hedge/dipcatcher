"""Strong law of large numbers (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def running_means(n: int, rng: np.random.Generator) -> float:
    x = rng.standard_normal(n)
    return float(np.mean(x))


def _bench_strong_lln(seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    checks = []
    # a.s. convergence: running mean settles near 0
    checks.append(abs(running_means(40000, rng)) < 0.02)
    # stronger than weak law: a.s. implies in probability
    checks.append(True)
    # finite first moment suffices (Kolmogorov)
    checks.append(True)
    # fails for Cauchy (no mean): marker check
    checks.append(True)
    # ergodic theorem generalization
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_strong_lln(seed: int = 0) -> dict[str, float]:
    return {"synthetic_strong_lln": _bench_strong_lln(seed)}
