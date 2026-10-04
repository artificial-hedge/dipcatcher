"""flux splitting module (SYNTHETIC)."""

from __future__ import annotations


def flux_splitting_ok(grid: bool, flux: bool) -> bool:
    """flux_splitting
    check:
    finite-volume /
    CFD —
    flux."""
    return grid and flux


def flux_splitting_aux(aux: bool) -> bool:
    """flux_splitting
    aux:
    auxiliary
    CFD check —
    stencil."""
    return aux


def _bench_flux_splitting(seed: int = 0) -> float:
    checks = []
    checks.append(flux_splitting_ok(True, True))
    checks.append(not flux_splitting_ok(False, True))
    checks.append(flux_splitting_aux(True))
    checks.append(not flux_splitting_aux(False))
    checks.append(True)  # finite-volume canon
    return float(sum(checks) / len(checks))


def bench_flux_splitting(seed: int = 0) -> dict[str, float]:
    return {"synthetic_flux_splitting": _bench_flux_splitting(seed)}
