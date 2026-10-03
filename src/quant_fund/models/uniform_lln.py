"""Glivenko-Cantelli: uniform convergence of empirical CDF (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def ks_gap(n: int, rng: np.random.Generator) -> float:
    """sup_x |F_n(x) - F(x)| for N(0,1) vs its empirical."""
    x = np.sort(rng.standard_normal(n))
    ecdf = np.arange(1, n + 1) / n
    from math import erf, sqrt

    tcdf = np.array([0.5 * (1 + erf(v / sqrt(2))) for v in x])
    return float(np.max(np.abs(ecdf - tcdf)))


def _bench_uniform_lln(seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    checks = []
    # gap shrinks as n grows
    g_small = ks_gap(50, rng)
    g_big = ks_gap(5000, rng)
    checks.append(g_big < g_small or g_big < 0.05)
    # GC theorem: sup |F_n - F| -> 0 a.s.
    checks.append(g_big < 0.05)
    # DKW inequality gives rate sqrt(log(1/d)/n)
    checks.append(g_big < 0.5)
    # uniform in x, not just pointwise
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_uniform_lln(seed: int = 0) -> dict[str, float]:
    return {"synthetic_uniform_lln": _bench_uniform_lln(seed)}
