"""Sylvester-matrix resultant and discriminant over QQ (synthetic).

Univariate polynomials as coefficient lists (highest degree first).
Resultant = det of the Sylvester matrix, computed with Bareiss
fraction-free elimination. Discriminant f via Res(f, f')/lead(f).
Verified against factored polynomials with known common roots.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb


def _sylvester(f: list[Fraction], g: list[Fraction]) -> list[list[Fraction]]:
    m, n = len(f) - 1, len(g) - 1
    size = m + n
    s = [[Fraction(0)] * size for _ in range(size)]
    for i in range(n):
        for j, c in enumerate(f):
            s[i][i + j] = c
    for i in range(m):
        for j, c in enumerate(g):
            s[n + i][i + j] = c
    return s


def _bareiss_det(a: list[list[Fraction]]) -> Fraction:
    n = len(a)
    m = [row[:] for row in a]
    sign = 1
    prev = Fraction(1)
    for k in range(n - 1):
        if m[k][k] == 0:
            piv = next((i for i in range(k + 1, n) if m[i][k] != 0), None)
            if piv is None:
                return Fraction(0)
            m[k], m[piv] = m[piv], m[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                m[i][j] = (m[i][j] * m[k][k] - m[i][k] * m[k][j]) / prev
        prev = m[k][k]
        for i in range(k + 1, n):
            m[i][k] = Fraction(0)
    return sign * m[n - 1][n - 1]


def resultant(f: list[Fraction], g: list[Fraction]) -> Fraction:
    return _bareiss_det(_sylvester(f, g))


def discriminant(f: list[Fraction]) -> Fraction:
    n = len(f) - 1
    fp = [c * (n - i) for i, c in enumerate(f)][:-1]
    sign = Fraction(-1) ** (n * (n - 1) // 2)
    return sign * resultant(f, fp) / f[0]


def bench_resultant(seed: int = 20261231 + 231) -> dict[str, float]:
    one = Fraction(1)
    # f = (x-1)(x-2) = x^2-3x+2 ; g = (x-2)(x-3) = x^2-5x+6 — share root 2
    f = [one, Fraction(-3), Fraction(2)]
    g = [one, Fraction(-5), Fraction(6)]
    r_shared = resultant(f, g)
    # h = x^2+1 — no common root with f
    h = [one, Fraction(0), one]
    r_coprime = resultant(f, h)
    # discriminant of x^2+bx+c must equal b^2-4c
    b_, c_ = Fraction(-3), Fraction(2)
    d = discriminant([one, b_, c_])
    d_true = b_ * b_ - 4 * c_
    # cubic discriminant of (x-1)(x-2)(x-3) = x^3-6x^2+11x-6 → 4
    cubic = [one, Fraction(-6), Fraction(11), Fraction(-6)]
    d3 = discriminant(cubic)
    agree = 0
    trials = 15
    import random

    rng = random.Random(seed)
    for _ in range(trials):
        r1, r2, r3, r4 = (rng.randint(-4, 4) for _ in range(4))
        p1 = [one, Fraction(-(r1 + r2)), Fraction(r1 * r2)]
        p2 = [one, Fraction(-(r3 + r4)), Fraction(r3 * r4)]
        res = resultant(p1, p2)
        expect_zero = bool({r1, r2} & {r3, r4})
        agree += int((res == 0) == expect_zero)
    return {
        "synthetic_res_shared": float(r_shared),
        "synthetic_res_coprime": float(r_coprime),
        "synthetic_disc": float(d),
        "synthetic_disc_true": float(d_true),
        "synthetic_disc3": float(d3),
        "synthetic_agree": float(agree / trials),
        "synthetic_comb_check": float(comb(4, 2) == 6),
    }
