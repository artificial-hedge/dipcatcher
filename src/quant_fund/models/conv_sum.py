"""Convolution of distributions: Bin+Bin, Poisson+Poisson, Normal+Normal (SYNTHETIC)."""

from __future__ import annotations

import math


def convolve(p: dict[int, float], q: dict[int, float]) -> dict[int, float]:
    out: dict[int, float] = {}
    for a, pa in p.items():
        for b, pb in q.items():
            out[a + b] = out.get(a + b, 0.0) + pa * pb
    return out


def binom_pmf(n: int, p: float) -> dict[int, float]:
    return {k: math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n + 1)}


def poisson_pmf(lam: float, kmax: int = 30) -> dict[int, float]:
    return {k: math.exp(-lam) * lam**k / math.factorial(k) for k in range(kmax)}


def _bench_conv_sum(seed: int = 0) -> float:
    checks = []
    b1 = binom_pmf(3, 0.5)
    b2 = binom_pmf(4, 0.5)
    conv = convolve(b1, b2)
    b7 = binom_pmf(7, 0.5)
    checks.append(all(abs(conv[k] - b7[k]) < 1e-9 for k in range(8)))
    checks.append(abs(sum(conv.values()) - 1.0) < 1e-9)
    # Poisson additivity (truncated tail)
    p3 = poisson_pmf(2.0, 25)
    p4 = poisson_pmf(3.0, 25)
    conv2 = convolve(p3, p4)
    p5 = poisson_pmf(5.0, 25)
    checks.append(all(abs(conv2[k] - p5[k]) < 1e-6 for k in range(20)))
    # deterministic shift
    d = convolve({0: 1.0}, b1)
    checks.append(all(abs(d[k] - b1[k]) < 1e-12 for k in b1))
    # expectation adds: E[B3] + E[B4] = E[B7]
    e1 = sum(k * v for k, v in b1.items())
    e2 = sum(k * v for k, v in b2.items())
    checks.append(abs(e1 + e2 - 3.5) < 1e-9)
    return float(sum(checks) / len(checks))


def bench_conv_sum(seed: int = 0) -> dict[str, float]:
    return {"synthetic_conv_sum": _bench_conv_sum(seed)}
