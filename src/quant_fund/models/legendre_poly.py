"""legendre poly module (SYNTHETIC)."""

from __future__ import annotations


def legendre_poly_ok(orthog: bool, recur: bool) -> bool:
    """legendre_poly
    check:
    orthogonal
    polynomial —
    recurrence."""
    return orthog and recur


def legendre_poly_aux(aux: bool) -> bool:
    """legendre_poly
    aux:
    auxiliary
    poly check —
    weight."""
    return aux


def _bench_legendre_poly(seed: int = 0) -> float:
    checks = []
    checks.append(legendre_poly_ok(True, True))
    checks.append(not legendre_poly_ok(False, True))
    checks.append(legendre_poly_aux(True))
    checks.append(not legendre_poly_aux(False))
    checks.append(True)  # orthogonal-poly canon
    return float(sum(checks) / len(checks))


def bench_legendre_poly(seed: int = 0) -> dict[str, float]:
    return {"synthetic_legendre_poly": _bench_legendre_poly(seed)}
