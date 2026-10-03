"""Newton divided-difference interpolation over QQ (synthetic).

Builds the exact interpolating polynomial through (x_i, y_i) in
Newton form via divided differences; verified by evaluating back at
the nodes and against a Lagrange oracle computed independently.
"""

from __future__ import annotations

from fractions import Fraction


def divided_diffs(xs: list[Fraction], ys: list[Fraction]) -> list[Fraction]:
    n = len(xs)
    coef = ys[:]
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (xs[i] - xs[i - j])
    return coef


def newton_eval(coef: list[Fraction], xs: list[Fraction], x: Fraction) -> Fraction:
    out = coef[-1]
    for i in range(len(coef) - 2, -1, -1):
        out = out * (x - xs[i]) + coef[i]
    return out


def lagrange_eval(xs: list[Fraction], ys: list[Fraction], x: Fraction) -> Fraction:
    out = Fraction(0)
    n = len(xs)
    for i in range(n):
        term = ys[i]
        for j in range(n):
            if j != i:
                term *= (x - xs[j]) / (xs[i] - xs[j])
        out += term
    return out


def bench_newton_interp(seed: int = 20261231 + 235) -> dict[str, float]:
    import random

    rng = random.Random(seed)
    agree = 0
    max_err_resid = Fraction(0)
    trials = 20
    for _ in range(trials):
        n = rng.randint(3, 6)
        xs = [Fraction(x) for x in rng.sample(range(-8, 9), n)]
        ys = [Fraction(rng.randint(-9, 9)) for _ in range(n)]
        coef = divided_diffs(xs, ys)
        # recovery at nodes
        resid = max(abs(newton_eval(coef, xs, xi) - yi) for xi, yi in zip(xs, ys, strict=True))
        max_err_resid = max(max_err_resid, resid)
        # agree with Lagrange at a fresh point
        xq = Fraction(rng.randint(-20, 20), 3)
        agree += int(newton_eval(coef, xs, xq) == lagrange_eval(xs, ys, xq))
    # degree check: points on a quadratic must give degree <= 2
    xs = [Fraction(i) for i in range(5)]
    ys = [x * x + 2 * x + 1 for x in xs]
    coef = divided_diffs(xs, ys)
    deg2 = all(c == 0 for c in coef[3:])
    return {
        "synthetic_agree": float(agree / trials),
        "synthetic_max_resid": float(max_err_resid),
        "synthetic_deg2_exact": float(deg2),
        "synthetic_trials": float(trials),
    }
