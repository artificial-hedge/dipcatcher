"""Glivenko-Cantelli: uniform convergence of empirical CDF (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def ks_gap(n: int, rng: np.random.Generator, x: np.ndarray | None = None) -> float:
    """sup_x |F_n(x) - F(x)| for N(0,1) vs its empirical."""
    if x is None:
        x = rng.standard_normal(n)
    x = np.sort(np.asarray(x, dtype=float))
    ecdf = np.arange(1, x.size + 1) / x.size
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
    # uniform in x, not just pointwise: the sup must agree with the
    # two-sided (left-continuous) KS convention on the SAME sample — the
    # two differ by at most one ecdf step (1/n) at the argmax.
    sample = rng.standard_normal(600)
    xs = np.sort(sample)
    e1 = np.arange(1, xs.size + 1) / xs.size
    from math import erf, sqrt

    tc = np.array([0.5 * (1 + erf(v / sqrt(2))) for v in xs])
    two_sided = float(max((e1 - tc).max(), (tc - (e1 - 1.0 / xs.size)).max()))
    one_sided = ks_gap(xs.size, rng, xs)
    checks.append(0.0 <= two_sided - one_sided <= 1.0 / xs.size + 1e-12)
    return float(sum(checks) / len(checks))


def bench_uniform_lln(seed: int = 0) -> dict[str, float]:
    return {"synthetic_uniform_lln": _bench_uniform_lln(seed)}
