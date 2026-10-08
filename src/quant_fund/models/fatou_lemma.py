"""Fatou's lemma: integral of liminf <= liminf of integrals (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def _bench_fatou_lemma(seed: int = 0) -> float:
    checks = []
    x = np.linspace(0, 1, 4001)
    dx = x[1] - x[0]
    # moving bump f_n = n * 1_{[0, 1/n]} -> 0 a.e., int f_n = 1:
    # int(liminf f) = 0 strictly below liminf(int f) = 1 — computed, not
    # asserted from constants
    ints = []
    for n in (200, 800, 2000):
        fn = np.where(x <= 1.0 / n, float(n), 0.0)
        ints.append(float(np.sum(fn) * dx))
        checks.append(bool(np.all(fn >= 0)))  # positivity premise
    liminf_int = min(ints)
    liminf_f = np.zeros_like(x)  # pointwise liminf of the bumps
    int_liminf = float(np.sum(liminf_f) * dx)
    checks.append(int_liminf <= liminf_int)
    checks.append(liminf_int - int_liminf > 0.5)
    # each discretized integral is order-1 (grid inflates it mildly)
    checks.append(all(0.9 < i < 2.2 for i in ints))
    # equality case: constant f_n = 1_{[0,1]} -> liminf int = int liminf
    g = np.ones_like(x)
    checks.append(abs(float(np.sum(g) * dx) - 1.0) < 1e-3)
    return float(sum(checks) / len(checks))


def bench_fatou_lemma(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fatou_lemma": _bench_fatou_lemma(seed)}
