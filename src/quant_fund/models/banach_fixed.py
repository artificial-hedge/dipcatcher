"""Banach contraction mapping theorem: iterate to unique fixed point (SYNTHETIC)."""

from __future__ import annotations

import math


def iterate(f, x0: float, n: int) -> list[float]:
    xs = [x0]
    for _ in range(n):
        xs.append(f(xs[-1]))
    return xs


def fixed_point(f, x0: float = 0.0, tol: float = 1e-12, maxit: int = 10_000) -> float:
    x = x0
    for _ in range(maxit):
        nx = f(x)
        if abs(nx - x) < tol:
            return float(nx)
        x = nx
    return float(x)


def lipschitz_const(f, lo: float, hi: float, steps: int = 200) -> float:
    xs = [lo + (hi - lo) * i / steps for i in range(steps + 1)]
    best = 0.0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            r = abs(f(xs[i]) - f(xs[j])) / abs(xs[i] - xs[j])
            best = max(best, r)
    return best


def _bench_banach_fixed(seed: int = 0) -> float:
    checks = []
    # f(x) = cos(x) on [0,1], contraction q~sin(1)
    f = math.cos
    fp = fixed_point(f, 0.5)
    checks.append(abs(f(fp) - fp) < 1e-10)
    q = lipschitz_const(f, 0.0, 1.0)
    checks.append(q < 0.85)
    # iterate converges geometrically
    xs = iterate(f, 0.5, 300)
    checks.append(abs(xs[-1] - fp) < 1e-6)
    # map [0,1] -> [0,1]
    checks.append(all(0.0 <= f(t) <= 1.0 for t in (0.0, 0.5, 1.0)))
    # sqrt(2) via Newton-like contraction g(x)=(x+2/x)/2
    g = lambda x: 0.5 * (x + 2.0 / x)  # noqa: E731
    fp2 = fixed_point(g, 1.0)
    checks.append(abs(fp2 * fp2 - 2.0) < 1e-10)
    checks.append(abs(g(fp2) - fp2) < 1e-10)
    return float(sum(checks) / len(checks))


def bench_banach_fixed(seed: int = 0) -> dict[str, float]:
    return {"synthetic_banach_fixed": _bench_banach_fixed(seed)}
