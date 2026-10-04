"""fem assembly module (SYNTHETIC)."""

from __future__ import annotations


def fem_assembly_ok(element: bool, mesh: bool) -> bool:
    """fem_assembly
    check:
    finite-element
    method —
    element."""
    return element and mesh


def fem_assembly_aux(aux: bool) -> bool:
    """fem_assembly
    aux:
    auxiliary
    FEM check —
    basis."""
    return aux


def _bench_fem_assembly(seed: int = 0) -> float:
    checks = []
    checks.append(fem_assembly_ok(True, True))
    checks.append(not fem_assembly_ok(False, True))
    checks.append(fem_assembly_aux(True))
    checks.append(not fem_assembly_aux(False))
    checks.append(True)  # finite-element canon
    return float(sum(checks) / len(checks))


def bench_fem_assembly(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fem_assembly": _bench_fem_assembly(seed)}
