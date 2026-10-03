"""Finite differences and Newton forward formula (SYNTHETIC)."""

from __future__ import annotations

import math


def diffs(vals: list[float]) -> list[float]:
    """Forward difference Delta f."""
    return [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]


def nth_diff(vals: list[float], n: int) -> list[float]:
    """Delta^n f."""
    cur = vals[:]
    for _ in range(n):
        cur = diffs(cur)
    return cur


def newton_eval(vals: list[float], x: float) -> float:
    """Newton forward interpolation at fractional x using binomial
    coefficients: f(x) = sum_k C(x, k) Delta^k f(0)."""
    total = 0.0
    for k in range(len(vals)):
        coeff = 1.0
        for j in range(k):
            coeff *= (x - j) / (j + 1)
        total += coeff * nth_diff(vals, k)[0]
    return total


def _bench_finite_difference(seed: int = 0) -> float:
    checks = []
    # Delta^3 of n^3 is constant 6
    cubes = [float(n**3) for n in range(6)]
    checks.append(all(abs(v - 6.0) < 1e-9 for v in nth_diff(cubes, 3)))
    # Delta^4 of cubic is zero
    checks.append(all(v == 0.0 for v in nth_diff(cubes, 4)))
    # Delta of arithmetic progression is constant difference
    checks.append(nth_diff([2 * n + 1 for n in range(5)], 1) == [2.0] * 4)
    # Newton formula recovers f(2.5) for f(n)=n^2 at nodes 0..3
    vals = [float(n * n) for n in range(4)]
    checks.append(abs(newton_eval(vals, 2.5) - 6.25) < 1e-9)
    # Newton at integer nodes equals tabulated values
    checks.append(abs(newton_eval(vals, 3.0) - 9.0) < 1e-9)
    # Delta^k f(0) for f(n)=C(n,k) is 1
    vals2 = [float(math.comb(n, 3)) for n in range(6)]
    checks.append(nth_diff(vals2, 3)[0] == 1.0)
    return float(sum(checks) / len(checks))


def bench_finite_difference(seed: int = 0) -> dict[str, float]:
    return {"synthetic_finite_difference": _bench_finite_difference(seed)}
