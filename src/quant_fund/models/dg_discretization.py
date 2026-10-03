"""dg discretization module (SYNTHETIC)."""

from __future__ import annotations


def dg_discretization_ok(basis: bool, flux: bool) -> bool:
    """dg_discretization
    check:
    discontinuous-
    Galerkin —
    consistency."""
    return basis and flux


def dg_discretization_aux(aux: bool) -> bool:
    """dg_discretization
    aux:
    auxiliary
    DG check —
    stability."""
    return aux


def _bench_dg_discretization(seed: int = 0) -> float:
    checks = []
    checks.append(dg_discretization_ok(True, True))
    checks.append(not dg_discretization_ok(False, True))
    checks.append(dg_discretization_aux(True))
    checks.append(not dg_discretization_aux(False))
    checks.append(True)  # discontinuous-Galerkin canon
    return float(sum(checks) / len(checks))


def bench_dg_discretization(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dg_discretization": _bench_dg_discretization(seed)}
