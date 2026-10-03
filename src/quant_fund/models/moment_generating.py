"""Moment generating functions: normal + Bernoulli/binomial (SYNTHETIC)."""

from __future__ import annotations

import math


def mgf_normal(mu: float, sigma2: float, t: float) -> float:
    return math.exp(mu * t + 0.5 * sigma2 * t * t)


def mgf_bernoulli(p: float, t: float) -> float:
    return 1 - p + p * math.exp(t)


def mgf_binomial(n: int, p: float, t: float) -> float:
    return mgf_bernoulli(p, t) ** n


def sample_mgf(samples: list[float], t: float) -> float:
    return sum(math.exp(t * s) for s in samples) / len(samples)


def _bench_moment_generating(seed: int = 0) -> float:
    checks = []
    import numpy as np

    # MGF derivative at 0 = mean: numerical check
    h = 1e-5
    checks.append(abs((mgf_normal(1.5, 2.0, h) - mgf_normal(1.5, 2.0, -h)) / (2 * h) - 1.5) < 1e-4)
    # second derivative = E[X^2] = mu^2 + sigma^2
    d2 = (mgf_normal(1.5, 2.0, h) - 2 + mgf_normal(1.5, 2.0, -h)) / (h * h)
    checks.append(abs(d2 - (1.5**2 + 2.0)) < 0.01)
    # binomial mgf at 0 = 1
    checks.append(mgf_binomial(5, 0.3, 0.0) == 1.0)
    # empirical MGF of samples ~ analytic
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, 200_000)
    emp = sample_mgf(list(x), 0.2)
    checks.append(abs(emp - mgf_normal(0.0, 1.0, 0.2)) / mgf_normal(0.0, 1.0, 0.2) < 0.02)
    # MGF of sum = product (independence): Bin(2) = Ber * Ber
    checks.append(abs(mgf_binomial(2, 0.3, 0.5) - mgf_bernoulli(0.3, 0.5) ** 2) < 1e-12)
    return float(sum(checks) / len(checks))


def bench_moment_generating(seed: int = 0) -> dict[str, float]:
    return {"synthetic_moment_generating": _bench_moment_generating(seed)}
