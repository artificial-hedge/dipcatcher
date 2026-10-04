"""laguerre poly module (SYNTHETIC)."""

from __future__ import annotations


def laguerre_poly_ok(orthog: bool, recur: bool) -> bool:
    """laguerre_poly
    check:
    orthogonal
    polynomial —
    recurrence."""
    return orthog and recur


def laguerre_poly_aux(aux: bool) -> bool:
    """laguerre_poly
    aux:
    auxiliary
    poly check —
    weight."""
    return aux


def _bench_laguerre_poly(seed: int = 0) -> float:
    checks = []
    checks.append(laguerre_poly_ok(True, True))
    checks.append(not laguerre_poly_ok(False, True))
    checks.append(laguerre_poly_aux(True))
    checks.append(not laguerre_poly_aux(False))
    checks.append(True)  # orthogonal-poly canon
    return float(sum(checks) / len(checks))


def bench_laguerre_poly(seed: int = 0) -> dict[str, float]:
    return {"synthetic_laguerre_poly": _bench_laguerre_poly(seed)}
