"""Gambler's ruin: hitting probabilities + expected duration (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def ruin_prob(i: int, n: int, p: float = 0.5) -> float:
    """P(hit n before 0 starting at i) for symmetric walk = i/n."""
    return i / n


def expected_duration(i: int, n: int) -> int:
    """E[T] for symmetric walk = i*(n-i)."""
    return i * (n - i)


def sim_ruin(i: int, n: int, rng: np.random.Generator, max_steps: int = 100_000) -> bool:
    x = i
    steps = rng.choice([-1, 1], max_steps)
    for s in steps:
        x += s
        if x == 0:
            return False
        if x == n:
            return True
    return x > n // 2


def _bench_gambler_ruin(seed: int = 0) -> float:
    checks = []
    n = 10
    checks.append(abs(ruin_prob(3, n) - 0.3) < 1e-9)
    checks.append(abs(ruin_prob(n, n) - 1.0) < 1e-9)
    checks.append(expected_duration(5, n) == 25)
    # Monte Carlo validation
    rng = np.random.default_rng(seed)
    wins = float(np.mean([sim_ruin(3, n, rng) for _ in range(20_000)]))
    checks.append(bool(abs(wins - 0.3) < 0.02))
    # symmetry: ruin prob of i = 1 - ruin prob of n-i (for hitting n vs 0 swap)
    checks.append(abs(ruin_prob(3, n) + ruin_prob(7, n) - 1.0) < 1e-9)
    # boundary: E[T | start at 1, n=2] = 1
    checks.append(expected_duration(1, 2) == 1)
    return float(sum(checks) / len(checks))


def bench_gambler_ruin(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gambler_ruin": _bench_gambler_ruin(seed)}
