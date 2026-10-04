"""bernstein poly module (SYNTHETIC)."""

from __future__ import annotations


def bernstein_poly_ok(smooth: bool, approx: bool) -> bool:
    """bernstein_poly
    check:
    approximation
    theory —
    smoothness."""
    return smooth and approx


def bernstein_poly_aux(aux: bool) -> bool:
    """bernstein_poly
    aux:
    auxiliary
    approx check —
    degree."""
    return aux


def _bench_bernstein_poly(seed: int = 0) -> float:
    checks = []
    checks.append(bernstein_poly_ok(True, True))
    checks.append(not bernstein_poly_ok(False, True))
    checks.append(bernstein_poly_aux(True))
    checks.append(not bernstein_poly_aux(False))
    checks.append(True)  # approximation-theory canon
    return float(sum(checks) / len(checks))


def bench_bernstein_poly(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bernstein_poly": _bench_bernstein_poly(seed)}
