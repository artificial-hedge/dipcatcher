"""Continued fractions and Pell equations (synthetic).

CF expansion of quadratic surds (exact integer arithmetic on
(a + b·√d)/c forms), convergents, and the minimal solution to
Pell x² - d·y² = 1 via the period of √d. Verified by plugging
solutions back and comparing convergent error vs 1/q² bound.
"""

from __future__ import annotations

from math import isqrt


def sqrt_cf(d: int) -> tuple[int, list[int]]:
    """CF of sqrt(d): a0 plus the repeating period."""
    a0 = isqrt(d)
    if a0 * a0 == d:
        return a0, []
    period: list[int] = []
    m, dn, a = 0, 1, a0
    while True:
        m = dn * a - m
        dn = (d - m * m) // dn
        a = (a0 + m) // dn
        period.append(a)
        if a == 2 * a0:
            break
    return a0, period


def convergent(cf0: int, period: list[int], k: int) -> tuple[int, int]:
    """k-th convergent p/q of [a0; a1, a2, ...] (period repeats)."""
    seq = [cf0] + [period[i % len(period)] for i in range(k)] if period else [cf0]
    p, q = 1, 0
    for a in reversed(seq):
        p, q = a * p + q, p
    return p, q


def pell_min(d: int) -> tuple[int, int]:
    """Minimal x,y > 0 with x² - d·y² = 1 (d nonsquare)."""
    a0, period = sqrt_cf(d)
    r = len(period)
    # minimal solution is convergent at index r-1 (r even) or 2r-1 (r odd)
    k = r - 1 if r % 2 == 0 else 2 * r - 1
    return convergent(a0, period, k)


def bench_continued_fraction(seed: int = 20261231 + 243) -> dict[str, float]:
    # classic: sqrt(2) = [1; 2,2,...]; Pell d=2 → (3,2); d=3 → (2,1)
    a0, period = sqrt_cf(2)
    ok_cf = a0 == 1 and period == [2]
    x2, y2 = pell_min(2)
    x3, y3 = pell_min(3)
    ok_pell = x2 * x2 - 2 * y2 * y2 == 1 and x3 * x3 - 3 * y3 * y3 == 1
    x23, y23 = pell_min(23)  # famous non-trivial: (24,5)
    ok_23 = x23 * x23 - 23 * y23 * y23 == 1
    # convergent error bound: |√d - p/q| < 1/q²
    import math

    bounds_ok = True
    for d in (2, 3, 5, 7):
        a0d, per = sqrt_cf(d)
        for k in range(1, 5):
            p, q = convergent(a0d, per, k)
            if abs(math.sqrt(d) - p / q) >= 1.0 / (q * q) + 1e-12:
                bounds_ok = False
    return {
        "synthetic_cf_sqrt2": float(ok_cf),
        "synthetic_pell_2": float((x2, y2) == (3, 2)),
        "synthetic_pell_ok": float(ok_pell and ok_23),
        "synthetic_pell23": float((x23, y23) == (24, 5)),
        "synthetic_conv_bound": float(bounds_ok),
    }
