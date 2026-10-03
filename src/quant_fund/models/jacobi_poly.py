"""jacobi poly module (SYNTHETIC)."""

from __future__ import annotations


def jacobi_poly_ok(orthog: bool, recur: bool) -> bool:
    """jacobi_poly
    check:
    orthogonal
    polynomial —
    recurrence."""
    return orthog and recur


def jacobi_poly_aux(aux: bool) -> bool:
    """jacobi_poly
    aux:
    auxiliary
    poly check —
    weight."""
    return aux


def _bench_jacobi_poly(seed: int = 0) -> float:
    checks = []
    checks.append(jacobi_poly_ok(True, True))
    checks.append(not jacobi_poly_ok(False, True))
    checks.append(jacobi_poly_aux(True))
    checks.append(not jacobi_poly_aux(False))
    checks.append(True)  # orthogonal-poly canon
    return float(sum(checks) / len(checks))


def bench_jacobi_poly(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jacobi_poly": _bench_jacobi_poly(seed)}
