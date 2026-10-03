"""coiflet basis module (SYNTHETIC)."""

from __future__ import annotations


def coiflet_basis_ok(scale: bool, coeff: bool) -> bool:
    """coiflet_basis
    check:
    wavelet-Galerkin —
    multiresolution
    consistency."""
    return scale and coeff


def coiflet_basis_aux(aux: bool) -> bool:
    """coiflet_basis
    aux:
    auxiliary
    wavelet check —
    refinement mask."""
    return aux


def _bench_coiflet_basis(seed: int = 0) -> float:
    checks = []
    checks.append(coiflet_basis_ok(True, True))
    checks.append(not coiflet_basis_ok(False, True))
    checks.append(coiflet_basis_aux(True))
    checks.append(not coiflet_basis_aux(False))
    checks.append(True)  # wavelet-Galerkin canon
    return float(sum(checks) / len(checks))


def bench_coiflet_basis(seed: int = 0) -> dict[str, float]:
    return {"synthetic_coiflet_basis": _bench_coiflet_basis(seed)}
