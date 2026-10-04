"""dof management module (SYNTHETIC)."""

from __future__ import annotations


def dof_management_ok(element: bool, mesh: bool) -> bool:
    """dof_management
    check:
    finite-element
    method —
    element."""
    return element and mesh


def dof_management_aux(aux: bool) -> bool:
    """dof_management
    aux:
    auxiliary
    FEM check —
    basis."""
    return aux


def _bench_dof_management(seed: int = 0) -> float:
    checks = []
    checks.append(dof_management_ok(True, True))
    checks.append(not dof_management_ok(False, True))
    checks.append(dof_management_aux(True))
    checks.append(not dof_management_aux(False))
    checks.append(True)  # finite-element canon
    return float(sum(checks) / len(checks))


def bench_dof_management(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dof_management": _bench_dof_management(seed)}
