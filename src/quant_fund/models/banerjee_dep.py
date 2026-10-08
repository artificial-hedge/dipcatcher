"""Data-dependence tests for affine array accesses (GCD + Banerjee bounds) (SYNTHETIC).

An access is (coeff, const) reading/writing a[i0 + c1*i1 + ... + ck*ik]
inside a loop nest with per-dim bounds (lo, hi). A dependence exists iff
f(i_w) = g(i_r) has an integer solution within both domains: GCD test is
necessary, Banerjee bounds test is sufficient-by-interval — both
conservative, exact on separable cases.
"""

from __future__ import annotations

import math
from fractions import Fraction

_SEED = 20261231 + 1030

Access = tuple[list[int], int]
Bounds = list[tuple[int, int]]


def _equation(f: Access, g: Access, bounds: Bounds) -> tuple[list[int], int, Bounds]:
    """f(i_w)=g(i_r) over independent i_w, i_r: coeffs are f's coefs on i_w
    and negated g's coefs on i_r; bounds duplicated per index vector."""
    coeffs = list(f[0]) + [-b for b in g[0]]
    rhs = g[1] - f[1]
    return coeffs, rhs, list(bounds) + list(bounds)


def gcd_test(f: Access, g: Access) -> bool:
    """Necessary condition: gcd of the equation coefficients divides rhs.
    Returns True iff a solution may exist (does NOT rule out dep)."""
    coeffs, rhs, _ = _equation(f, g, [])
    g_ = 0
    for c in coeffs:
        g_ = math.gcd(g_, abs(c))
    return g_ == 0 and rhs == 0 or g_ != 0 and rhs % g_ == 0


def banerjee_test(f: Access, g: Access, bounds: Bounds) -> bool:
    """Interval test on the equation a·x = rhs over the (i_w, i_r) box:
    solvable only if rhs lies between the form's min and max."""
    coeffs, rhs, bnds = _equation(f, g, bounds)
    lo_sum, hi_sum = 0, 0
    for c, (lo, hi) in zip(coeffs, bnds, strict=True):
        if c >= 0:
            lo_sum += c * lo
            hi_sum += c * hi
        else:
            lo_sum += c * hi
            hi_sum += c * lo
    return lo_sum <= rhs <= hi_sum


def dependence(f: Access, g: Access, bounds: Bounds) -> bool:
    """Conservative dependence oracle: both tests must admit a solution."""
    return gcd_test(f, g) and banerjee_test(f, g, bounds)


def direction(f: Access, g: Access, bounds: Bounds, dim: int) -> Fraction | None:
    """Dependence distance in dim: exact single-index case f=g=c*i+k."""
    a = f[0][dim]
    b = g[0][dim]
    if a == b and f[1] == g[1]:
        return Fraction(0)
    if a == 0 and b == 0:
        return Fraction(0) if f[1] == g[1] else None
    if a != 0 and b == 0:
        i_eq = Fraction(g[1] - f[1], a)
        return i_eq if f[1] + a * i_eq == g[1] else None
    return None


def bench_banerjee_dep(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    # A[i] = ... then ... = A[i]: same access -> dep
    checks.append(dependence(([1], 0), ([1], 0), [(0, 10)]))
    # A[i] w, A[i+1] r: i_w - i_r = 1 -> dep (cross-iteration)
    checks.append(dependence(([1], 0), ([1], 1), [(0, 10)]))
    # A[i] w, A[i+10] r over i in 0..9: needs i_w - i_r = 10 but max |diff|=9 -> Banerjee rules out
    checks.append(not dependence(([1], 0), ([1], 10), [(0, 9)]))
    # A[2i] w, A[2i+1] r: gcd(2,2)=2, rhs=1 odd -> GCD rules out
    checks.append(not dependence(([2], 0), ([2], 1), [(0, 20)]))
    # 2D: A[i,j] w vs A[i,j] r -> dep; A[i,j] vs A[j,i] shares indices -> possible
    checks.append(dependence(([1, 0], 0), ([1, 0], 0), [(0, 5), (0, 5)]))
    return {"synthetic_banerjee_dep": float(sum(checks)) / len(checks)}
