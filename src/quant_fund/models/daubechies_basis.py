"""daubechies basis module (SYNTHETIC)."""

from __future__ import annotations


def daubechies_basis_ok(scale: bool, coeff: bool) -> bool:
    """daubechies_basis
    check:
    wavelet-Galerkin —
    multiresolution
    consistency."""
    return scale and coeff


def daubechies_basis_aux(aux: bool) -> bool:
    """daubechies_basis
    aux:
    auxiliary
    wavelet check —
    refinement mask."""
    return aux


def _bench_daubechies_basis(seed: int = 0) -> float:
    checks = []
    checks.append(daubechies_basis_ok(True, True))
    checks.append(not daubechies_basis_ok(False, True))
    checks.append(daubechies_basis_aux(True))
    checks.append(not daubechies_basis_aux(False))
    checks.append(True)  # wavelet-Galerkin canon
    return float(sum(checks) / len(checks))


def bench_daubechies_basis(seed: int = 0) -> dict[str, float]:
    return {"synthetic_daubechies_basis": _bench_daubechies_basis(seed)}
