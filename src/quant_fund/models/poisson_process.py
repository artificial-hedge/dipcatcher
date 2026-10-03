"""Poisson process: exponential interarrivals, Poisson counts, thinning (SYNTHETIC)."""

from __future__ import annotations

import math

import numpy as np


def sim_poisson_process(lam: float, t: float, rng: np.random.Generator) -> int:
    return int(rng.poisson(lam * t))


def exp_interarrival_mean(lam: float, n: int, rng: np.random.Generator) -> float:
    return float(np.mean(rng.exponential(1.0 / lam, n)))


def _bench_poisson_process(seed: int = 0) -> float:
    checks = []
    rng = np.random.default_rng(seed)
    # interarrival mean = 1/lambda
    checks.append(abs(exp_interarrival_mean(2.0, 200_000, rng) - 0.5) < 0.01)
    # counts: N(1) ~ Poisson(lam)
    counts = np.array([sim_poisson_process(2.0, 1.0, rng) for _ in range(50_000)])
    checks.append(abs(float(np.mean(counts)) - 2.0) < 0.03)
    checks.append(abs(float(np.var(counts)) - 2.0) < 0.1)
    # thinning: marking each event Bernoulli(0.3) -> Poisson(0.6)
    thin = np.array([rng.binomial(c, 0.3) for c in counts])
    checks.append(abs(float(np.mean(thin)) - 0.6) < 0.02)
    # superposition: Pois(2)+Pois(3) ~ Pois(5)
    c5 = np.array(
        [
            sim_poisson_process(2.0, 1.0, rng) + sim_poisson_process(3.0, 1.0, rng)
            for _ in range(30_000)
        ]
    )
    checks.append(abs(float(np.mean(c5)) - 5.0) < 0.05)
    # memorylessness: P(X > s+t | X > s) = P(X > t)
    x = rng.exponential(1.0, 200_000)
    s0, t0 = 0.5, 1.0
    cond = float(np.mean(x[x > s0] > s0 + t0))
    checks.append(abs(cond - math.exp(-t0)) < 0.02)
    return float(sum(checks) / len(checks))


def bench_poisson_process(seed: int = 0) -> dict[str, float]:
    return {"synthetic_poisson_process": _bench_poisson_process(seed)}
