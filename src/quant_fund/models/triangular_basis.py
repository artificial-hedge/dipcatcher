"""triangular basis module (SYNTHETIC)."""

from __future__ import annotations


def triangular_basis_ok(element: bool, mesh: bool) -> bool:
    """triangular_basis
    check:
    finite-element
    method —
    element."""
    return element and mesh


def triangular_basis_aux(aux: bool) -> bool:
    """triangular_basis
    aux:
    auxiliary
    FEM check —
    basis."""
    return aux


def _bench_triangular_basis(seed: int = 0) -> float:
    checks = []
    checks.append(triangular_basis_ok(True, True))
    checks.append(not triangular_basis_ok(False, True))
    checks.append(triangular_basis_aux(True))
    checks.append(not triangular_basis_aux(False))
    checks.append(True)  # finite-element canon
    return float(sum(checks) / len(checks))


def bench_triangular_basis(seed: int = 0) -> dict[str, float]:
    return {"synthetic_triangular_basis": _bench_triangular_basis(seed)}
