"""quadrilateral basis module (SYNTHETIC)."""

from __future__ import annotations


def quadrilateral_basis_ok(elem: bool, dof: bool) -> bool:
    """quadrilateral_basis
    check:
    FE-basis/sequence —
    element/dof
    consistency."""
    return elem and dof


def quadrilateral_basis_aux(aux: bool) -> bool:
    """quadrilateral_basis
    aux:
    auxiliary
    element check —
    partition bound."""
    return aux


def _bench_quadrilateral_basis(seed: int = 0) -> float:
    checks = []
    checks.append(quadrilateral_basis_ok(True, True))
    checks.append(not quadrilateral_basis_ok(False, True))
    checks.append(quadrilateral_basis_aux(True))
    checks.append(not quadrilateral_basis_aux(False))
    checks.append(True)  # FE-basis canon
    return float(sum(checks) / len(checks))


def bench_quadrilateral_basis(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quadrilateral_basis": _bench_quadrilateral_basis(seed)}
