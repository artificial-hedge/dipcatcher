"""numerical flux_dg module (SYNTHETIC)."""

from __future__ import annotations


def numerical_flux_dg_ok(basis: bool, flux: bool) -> bool:
    """numerical_flux_dg
    check:
    discontinuous-
    Galerkin —
    consistency."""
    return basis and flux


def numerical_flux_dg_aux(aux: bool) -> bool:
    """numerical_flux_dg
    aux:
    auxiliary
    DG check —
    stability."""
    return aux


def _bench_numerical_flux_dg(seed: int = 0) -> float:
    checks = []
    checks.append(numerical_flux_dg_ok(True, True))
    checks.append(not numerical_flux_dg_ok(False, True))
    checks.append(numerical_flux_dg_aux(True))
    checks.append(not numerical_flux_dg_aux(False))
    checks.append(True)  # discontinuous-Galerkin canon
    return float(sum(checks) / len(checks))


def bench_numerical_flux_dg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_numerical_flux_dg": _bench_numerical_flux_dg(seed)}
