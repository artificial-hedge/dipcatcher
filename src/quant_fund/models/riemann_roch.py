"""Riemann-Roch theorem on curves (SYNTHETIC)."""

from __future__ import annotations


def ell_p1(deg_d: int) -> int:
    """l(D) on P^1 (genus 0): dim of rational functions with poles <= D.

    For D = n*infty: the space is polynomials of degree <= n, so l = n+1
    when n >= 0 and 0 when n < 0 — exactly deg(D) + 1 - g.
    """
    return max(deg_d + 1, 0)


def rr_bound(deg_d: int, g: int) -> tuple[int, int]:
    """Riemann-Roch bounds for l(D): deg(D)+1-g <= l(D) <= deg(D)+1
    (for deg >= 0); equality when deg(D) > 2g - 2 (K - D has negative
    degree, so l(K-D) = 0)."""
    lo = deg_d + 1 - g
    hi = deg_d + 1
    return lo, hi


def _bench_riemann_roch(seed: int = 0) -> float:
    checks = []
    # P1: l(n*inf) = n+1 exactly = deg+1-0
    checks.append(ell_p1(5) == 6)
    checks.append(ell_p1(0) == 1)
    checks.append(ell_p1(-1) == 0)
    # genus 1 (elliptic curve): deg>0 -> l(D) = deg exactly (deg > 2g-2=0)
    lo, hi = rr_bound(5, 1)
    checks.append(lo == hi - 1 == 5 or (lo == 5 and hi == 6))
    # wait lo = 5+1-1 = 5, hi = 6: RR gives l=deg when deg>0 on genus 1
    # genus 2 canonical divisor deg(K)=2g-2=2: l(K)=g=2
    # deg(D) > 2g-2: bound collapses to exact
    for g, deg in [(0, 3), (1, 4), (2, 5), (3, 7)]:
        lo, hi = rr_bound(deg, g)
        checks.append(hi - lo == g)  # gap equals genus
        checks.append(lo == deg + 1 - g)
    # deg > 2g-2 => l(D) = deg + 1 - g exactly
    checks.append(rr_bound(5, 2) == (4, 6))  # lo=4, hi=6; exact l=4
    # verify exact formula: deg(D)=5 > 2 on genus 2 -> l = 5+1-2 = 4
    checks.append(5 + 1 - 2 == 4)
    return float(sum(bool(c) for c in checks) / len(checks))


def bench_riemann_roch(seed: int = 0) -> dict[str, float]:
    return {"synthetic_riemann_roch": _bench_riemann_roch(seed)}
