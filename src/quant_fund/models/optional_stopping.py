"""Doob's optional stopping theorem on bounded stopping times (SYNTHIC MC) (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def srw_stopped(level: int, cap: int, rng: np.random.Generator) -> float:
    """SRW from 0 stopped at first hit of +-level or time cap."""
    x = 0
    for _ in range(cap):
        x += 1 if rng.random() < 0.5 else -1
        if abs(x) == level:
            break
    return float(x)


def _bench_optional_stopping(seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    checks = []
    # bounded stopping time -> E[X_T] = E[X_0] = 0
    draws = np.array([srw_stopped(3, 500, rng) for _ in range(40000)])
    checks.append(abs(float(np.mean(draws))) < 0.06)
    # hitting prob of +a before -b is b/(a+b) (gambler): E[X_T] = a p - b(1-p) = 0
    a_l, b_l = 2, 4
    hits = 0
    n = 30000
    for _ in range(n):
        x = 0
        while x != a_l and x != -b_l:
            x += 1 if rng.random() < 0.5 else -1
        hits += int(x == a_l)
    checks.append(abs(hits / n - b_l / (a_l + b_l)) < 0.02)

    # hitting time to +2 capped at C stays bounded -> E[S_T] = 0 for any C,
    # while E[min(T_hit, C)] grows ~ sqrt(C) -> the uncapped time is
    # unbounded and outside OST.
    def capped_hit(cap: int) -> tuple[float, float]:
        ts = []
        vals = []
        for _ in range(4000):
            x = 0
            t = 0
            while t < cap and x != 2:
                x += 1 if rng.random() < 0.5 else -1
                t += 1
            ts.append(float(t))
            vals.append(x)
        return float(np.mean(ts)), float(np.mean(vals))

    t200, v200 = capped_hit(200)
    t2000, v2000 = capped_hit(2000)
    checks.append(t2000 > 1.5 * t200)
    checks.append(abs(v200) < 0.2 and abs(v2000) < 0.2)
    # E[S_T^2] = E[T] for the bounded +-3 stopping time (S^2 - n martingale)
    checks.append(abs(float(np.var(draws)) - 9.0) < 9.0 * 0.2)
    return float(sum(checks) / len(checks))


def bench_optional_stopping(seed: int = 0) -> dict[str, float]:
    return {"synthetic_optional_stopping": _bench_optional_stopping(seed)}
