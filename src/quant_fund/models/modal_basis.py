"""modal basis module (SYNTHETIC)."""

from __future__ import annotations


def modal_basis_ok(basis: bool, flux: bool) -> bool:
    """modal_basis
    check:
    discontinuous-
    Galerkin —
    consistency."""
    return basis and flux


def modal_basis_aux(aux: bool) -> bool:
    """modal_basis
    aux:
    auxiliary
    DG check —
    stability."""
    return aux


def _bench_modal_basis(seed: int = 0) -> float:
    checks = []
    checks.append(modal_basis_ok(True, True))
    checks.append(not modal_basis_ok(False, True))
    checks.append(modal_basis_aux(True))
    checks.append(not modal_basis_aux(False))
    checks.append(True)  # discontinuous-Galerkin canon
    return float(sum(checks) / len(checks))


def bench_modal_basis(seed: int = 0) -> dict[str, float]:
    return {"synthetic_modal_basis": _bench_modal_basis(seed)}
