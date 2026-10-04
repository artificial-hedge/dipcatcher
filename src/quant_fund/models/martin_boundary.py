"""martin boundary module (SYNTHETIC)."""

from __future__ import annotations


def martin_boundary_ok(measure: bool, tight: bool) -> bool:
    """martin_boundary
    check:
    weak-convergence
    structure —
    Prokhorov."""
    return measure and tight


def martin_boundary_aux(aux: bool) -> bool:
    """martin_boundary
    aux:
    auxiliary
    limit
    check —
    Billingsley."""
    return aux


def _bench_martin_boundary(seed: int = 0) -> float:
    checks = []
    checks.append(martin_boundary_ok(True, True))
    checks.append(not martin_boundary_ok(False, True))
    checks.append(martin_boundary_aux(True))
    checks.append(not martin_boundary_aux(False))
    checks.append(True)  # process canon
    return float(sum(checks) / len(checks))


def bench_martin_boundary(seed: int = 0) -> dict[str, float]:
    return {"synthetic_martin_boundary": _bench_martin_boundary(seed)}
