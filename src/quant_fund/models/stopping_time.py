"""Optional stopping: E[X_T]=X_0 for bounded stopping times on fair games (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def stopping_payoff(rng: np.random.Generator, cap: int) -> float:
    """Random walk stopped when |X|=3 or t=cap (bounded stop)."""
    x = 0
    for _ in range(cap):
        x += int(rng.choice([-1, 1]))
        if abs(x) == 3:
            break
    return float(x)


def hit_time(rng: np.random.Generator, level: int, cap: int) -> int:
    x = 0
    for t in range(1, cap + 1):
        x += int(rng.choice([-1, 1]))
        if abs(x) == level:
            return t
    return cap


def _bench_stopping_time(seed: int = 0) -> float:
    checks = []
    rng = np.random.default_rng(seed)
    # bounded stopping preserves mean: E[X_T] = 0
    payoffs = np.array([stopping_payoff(rng, 100) for _ in range(60_000)])
    checks.append(abs(float(np.mean(payoffs))) < 0.05)
    # E[hit time to ±3] <= cap and around 9 (theoretical = 9 for ±a symmetric)
    times = np.array([hit_time(rng, 3, 10_000) for _ in range(20_000)])
    checks.append(abs(float(np.mean(times)) - 9.0) < 0.5)
    # first hit time has same parity as level: hitting 3 takes odd steps
    t = hit_time(rng, 3, 10_000)
    checks.append(t % 2 == 1)
    # Doob: stopped bounded martingale variance positive
    checks.append(float(np.var(payoffs)) > 0.5)
    # stopping at unbounded level on finite cap: payoff within bounds
    checks.append(all(abs(p) <= 3.0001 for p in payoffs[:1000]))
    return float(sum(checks) / len(checks))


def bench_stopping_time(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stopping_time": _bench_stopping_time(seed)}
