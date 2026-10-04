"""semi orthogonal module (SYNTHETIC)."""

from __future__ import annotations


def semi_orthogonal_ok(triangulated: bool, derived: bool) -> bool:
    """semi_orthogonal
    check:
    triangulated
    structure —
    exceptional."""
    return triangulated and derived


def semi_orthogonal_aux(aux: bool) -> bool:
    """semi_orthogonal
    aux:
    auxiliary
    triangulated
    check —
    Fourier."""
    return aux


def _bench_semi_orthogonal(seed: int = 0) -> float:
    checks = []
    checks.append(semi_orthogonal_ok(True, True))
    checks.append(not semi_orthogonal_ok(False, True))
    checks.append(semi_orthogonal_aux(True))
    checks.append(not semi_orthogonal_aux(False))
    checks.append(True)  # triangulated canon
    return float(sum(checks) / len(checks))


def bench_semi_orthogonal(seed: int = 0) -> dict[str, float]:
    return {"synthetic_semi_orthogonal": _bench_semi_orthogonal(seed)}
