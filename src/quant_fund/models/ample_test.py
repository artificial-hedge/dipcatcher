"""Ampleness via Nakai-Moishezon on surfaces (SYNTHETIC)."""

from __future__ import annotations


def nakai_surface(self_int: int, curve_intersections: list[int]) -> bool:
    """D ample on a surface iff D^2 > 0 and D.C > 0 for all curves C."""
    return self_int > 0 and all(c > 0 for c in curve_intersections)


def kodaira_vanishing_check(ample: bool, i: int) -> bool:
    """Kodaira: H^i(X, K + A) = 0 for i > 0 when A ample; model returns
    whether cohomology is FORCED to vanish (i > 0 and ample)."""
    return bool(ample and i > 0)


def _bench_ample_test(seed: int = 0) -> float:
    checks = []
    # O(1) on P2: H^2 = 1 > 0, meets every curve positively -> ample
    checks.append(nakai_surface(1, [1, 2, 3]))
    # O(1,0) on P1xP1: self-int 0 -> not ample (semi-ample only)
    checks.append(not nakai_surface(0, [1, 0]))
    # O(1,1) on P1xP1: self 2, meets rulings 1 each -> ample
    checks.append(nakai_surface(2, [1, 1]))
    # -1 curve on blowup of P2: self -1 -> negative -> not ample
    checks.append(not nakai_surface(-1, [-1]))
    # canonical + ample vanishes in positive degrees (Kodaira check on P2:
    # H^i(P2, O(K+3H)) = H^i(P2, O(0)) = 0 for i>0 -> ample H passes)
    checks.append(kodaira_vanishing_check(True, 1))
    checks.append(not kodaira_vanishing_check(False, 1))
    # degree-0 doesn't force vanishing
    checks.append(not kodaira_vanishing_check(True, 0))
    return float(sum(checks) / len(checks))


def bench_ample_test(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ample_test": _bench_ample_test(seed)}
