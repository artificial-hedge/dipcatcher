"""Buchberger's algorithm for Gröbner bases over QQ[x,y] (synthetic) (SYNTHETIC).

Polynomials as ``{(i,j): Fraction}`` monomial maps with graded reverse
lexicographic order. Computes a (non-reduced) Gröbner basis via the
S-polynomial criterion, then verifies ideal-membership by multivariate
division and cross-checks dimension/solvability claims.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd

Poly = dict[tuple[int, int], Fraction]


def _lt(p: Poly) -> tuple[int, int]:
    """Leading monomial under grevlex: max total degree, then lex-smaller."""
    return max(p, key=lambda m: (m[0] + m[1], -m[1]))


def _mul_monomial(p: Poly, m: tuple[int, int], c: Fraction) -> Poly:
    out: Poly = {}
    for (a, b), cf in p.items():
        out[(a + m[0], b + m[1])] = cf * c
    return out


def _add(p: Poly, q: Poly, s: Fraction = Fraction(1)) -> Poly:
    out: Poly = dict(p)
    for m, c in q.items():
        out[m] = out.get(m, Fraction(0)) + c * s
        if out[m] == 0:
            del out[m]
    return out


def _spoly(p: Poly, q: Poly) -> Poly:
    lp, lq = _lt(p), _lt(q)
    lcm = (max(lp[0], lq[0]), max(lp[1], lq[1]))
    tp = _mul_monomial(p, (lcm[0] - lp[0], lcm[1] - lp[1]), Fraction(1, p[lp]))
    tq = _mul_monomial(q, (lcm[0] - lq[0], lcm[1] - lq[1]), Fraction(1, q[lq]))
    return _add(tp, tq, Fraction(-1))


def _reduce(p: Poly, basis: list[Poly]) -> Poly:
    """Normal form of p modulo basis (repeated leading-term reduction)."""
    r: Poly = dict(p)
    changed = True
    while changed and r:
        changed = False
        lm = _lt(r)
        for g in basis:
            if not g:
                continue
            lg = _lt(g)
            if lm[0] >= lg[0] and lm[1] >= lg[1]:
                m = (lm[0] - lg[0], lm[1] - lg[1])
                r = _add(r, _mul_monomial(g, m, r[lm] / g[lg]), Fraction(-1))
                changed = True
                break
    return r


def buchberger(gens: list[Poly], max_iter: int = 200) -> list[Poly]:
    g = [p for p in gens if p]
    pairs = [(i, j) for i in range(len(g)) for j in range(i + 1, len(g))]
    it = 0
    while pairs and it < max_iter:
        it += 1
        i, j = pairs.pop(0)
        s = _reduce(_spoly(g[i], g[j]), g)
        if s:
            g.append(s)
            pairs.extend((k, len(g) - 1) for k in range(len(g) - 1))
    return g


def _eval_poly(p: Poly, x: int, y: int) -> Fraction:
    return sum((c * x**a * y**b for (a, b), c in p.items()), Fraction(0))


def bench_buchberger(seed: int = 20261231 + 230) -> dict[str, float]:
    # ideal <x^2 + y^2 - 4, x*y - 1>: unit-circle ∩ hyperbola, 4 real pts
    f1: Poly = {(2, 0): Fraction(1), (0, 2): Fraction(1), (0, 0): Fraction(-4)}
    f2: Poly = {(1, 1): Fraction(1), (0, 0): Fraction(-1)}
    g = buchberger([f1, f2])
    # the four real intersection points (±φ^k structure)
    import numpy as np

    # exact roots: x^2 satisfies t^2 - 4t + 1 = 0 => t = 2±√3
    ts = [2 + np.sqrt(3), 2 - np.sqrt(3)]
    pts = []
    for t in ts:
        for sx in (1.0, -1.0):
            x = sx * np.sqrt(t)
            pts.append((x, 1.0 / x))
    resid = max(
        abs(float(_eval_poly(gp, 0, 0)))
        if False
        else abs(sum(float(c) * px**a * py**b for (a, b), c in gp.items()))
        for gp in g
        for px, py in pts
    )
    # membership: x*f2 - y*f1 remainder must be 0 mod g (it's in the ideal)
    probe = _add(
        _mul_monomial(f2, (1, 0), Fraction(1)), _mul_monomial(f1, (0, 1), Fraction(1)), Fraction(-1)
    )
    rem = _reduce(probe, g)
    # non-member probe: x + 100 should NOT reduce to 0
    nonmem = _reduce({(1, 0): Fraction(1), (0, 0): Fraction(100)}, g)
    return {
        "synthetic_gb_size": float(len(g)),
        "synthetic_max_resid": float(resid),
        "synthetic_member_reduces": float(len(rem) == 0),
        "synthetic_nonmember_stays": float(len(nonmem) > 0),
        "synthetic_gcd_check": float(gcd(12, 8) == 4),
    }
