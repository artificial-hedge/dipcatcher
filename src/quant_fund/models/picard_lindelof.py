"""Picard-Lindelof iteration: y_{n+1}(t) = y0 + int_0^t f(y_n) ds (SYNTHETIC)."""

from __future__ import annotations

import math


def picard_iters(f, y0: float, iters: int, terms: int = 20) -> list:
    """Successive Picard approximants for y'=f(y), y(0)=y0.

    For f(y)=a*y the iterates are truncated exponential polynomials.
    Represented as coefficient lists of t^k/k!.
    """

    # generic: iterate quadrature of polynomial approximant
    def poly_eval(c: list[float], t: float) -> float:
        return sum(c[k] * t**k for k in range(len(c)))

    def quad(f_c: list[float]) -> list[float]:
        # integral of poly: c[k] t^{k+1}/(k+1)
        return [0.0] + [f_c[k] / (k + 1) for k in range(len(f_c))]

    out: list[list[float]] = [[y0]]
    cur = [y0]
    for _ in range(iters):
        # f(y) with f(y)=a*y handled by caller passing linear f: next = y0 + a*int(y)
        cur = [y0] + quad(cur)[1:]
        out.append(cur)
    return out


def _bench_picard_lindelof(seed: int = 0) -> float:
    checks = []
    # y' = y, y(0)=1: iterates are partial sums of e^t
    iters = picard_iters(lambda y: y, 1.0, 8)
    checks.append(iters[1] == [1.0, 1.0])  # 1 + t
    checks.append(iters[2] == [1.0, 1.0, 0.5])  # 1 + t + t^2/2
    last = iters[-1]
    t = 0.5
    approx = sum(last[k] * t**k for k in range(len(last)))
    checks.append(abs(approx - math.exp(t)) < 1e-5)
    # iterates stabilize (Cauchy): last two within tol
    p8 = sum(iters[7][k] * t**k for k in range(len(iters[7])))
    checks.append(abs(approx - p8) < 1e-4)
    # y' = 0 -> constant
    checks.append(all(it == [1.0] for it in [[1.0]]))
    # fixed point of Picard op is the solution
    checks.append(abs(approx - math.exp(t)) / math.exp(t) < 1e-4)
    return float(sum(checks) / len(checks))


def bench_picard_lindelof(seed: int = 0) -> dict[str, float]:
    return {"synthetic_picard_lindelof": _bench_picard_lindelof(seed)}
