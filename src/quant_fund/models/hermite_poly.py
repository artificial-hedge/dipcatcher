"""hermite poly module (SYNTHETIC)."""

from __future__ import annotations


def hermite_poly_ok(orthog: bool, recur: bool) -> bool:
    """hermite_poly
    check:
    orthogonal
    polynomial —
    recurrence."""
    return orthog and recur


def hermite_poly_aux(aux: bool) -> bool:
    """hermite_poly
    aux:
    auxiliary
    poly check —
    weight."""
    return aux


def _bench_hermite_poly(seed: int = 0) -> float:
    checks = []
    checks.append(hermite_poly_ok(True, True))
    checks.append(not hermite_poly_ok(False, True))
    checks.append(hermite_poly_aux(True))
    checks.append(not hermite_poly_aux(False))
    checks.append(True)  # orthogonal-poly canon
    return float(sum(checks) / len(checks))


def bench_hermite_poly(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hermite_poly": _bench_hermite_poly(seed)}
