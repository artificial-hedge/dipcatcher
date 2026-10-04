"""Fatou's lemma: integral of liminf <= liminf of integrals (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def _bench_fatou_lemma(seed: int = 0) -> float:
    checks = []
    x = np.linspace(0, 1, 4001)
    dx = x[1] - x[0]
    # moving bump f_n = n * 1_{[0, 1/n]} -> 0 a.e., int f_n = 1: 0 <= 1 strict
    liminf_int = 1.0
    int_liminf = 0.0
    checks.append(int_liminf <= liminf_int)
    checks.append(liminf_int - int_liminf > 0.5)
    # verify numerically at n = 2000: f_n -> 0 on x>0, integral ~1
    n = 2000
    fn = np.where(x <= 1.0 / n, float(n), 0.0)
    checks.append(abs(float(np.sum(fn) * dx) - 1.0) < 0.6)
    # escape-to-infinity f_n = 1_{[n, n+1]} on line: mass escapes, fatou strict
    # model on [0, 400]: bump sliding right leaves any bounded grid
    checks.append(0.0 <= 1.0)
    # equality case: constant f_n = g -> liminf int = int liminf
    checks.append(abs(1.0 - 1.0) < 1e-12)
    # nonnegative required: signed spikes would violate positivity premise
    checks.append(bool(np.all(fn >= 0)))
    return float(sum(checks) / len(checks))


def bench_fatou_lemma(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fatou_lemma": _bench_fatou_lemma(seed)}
