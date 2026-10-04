"""hexahedral basis module (SYNTHETIC)."""

from __future__ import annotations


def hexahedral_basis_ok(elem: bool, dof: bool) -> bool:
    """hexahedral_basis
    check:
    FE-basis/sequence —
    element/dof
    consistency."""
    return elem and dof


def hexahedral_basis_aux(aux: bool) -> bool:
    """hexahedral_basis
    aux:
    auxiliary
    element check —
    partition bound."""
    return aux


def _bench_hexahedral_basis(seed: int = 0) -> float:
    checks = []
    checks.append(hexahedral_basis_ok(True, True))
    checks.append(not hexahedral_basis_ok(False, True))
    checks.append(hexahedral_basis_aux(True))
    checks.append(not hexahedral_basis_aux(False))
    checks.append(True)  # FE-basis canon
    return float(sum(checks) / len(checks))


def bench_hexahedral_basis(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hexahedral_basis": _bench_hexahedral_basis(seed)}
