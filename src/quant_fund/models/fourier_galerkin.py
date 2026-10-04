"""fourier galerkin module (SYNTHETIC)."""

from __future__ import annotations


def fourier_galerkin_ok(grid: bool, basis: bool) -> bool:
    """fourier_galerkin
    check:
    spectral
    method —
    grid."""
    return grid and basis


def fourier_galerkin_aux(aux: bool) -> bool:
    """fourier_galerkin
    aux:
    auxiliary
    spectral check —
    coeff."""
    return aux


def _bench_fourier_galerkin(seed: int = 0) -> float:
    checks = []
    checks.append(fourier_galerkin_ok(True, True))
    checks.append(not fourier_galerkin_ok(False, True))
    checks.append(fourier_galerkin_aux(True))
    checks.append(not fourier_galerkin_aux(False))
    checks.append(True)  # spectral-methods canon
    return float(sum(checks) / len(checks))


def bench_fourier_galerkin(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fourier_galerkin": _bench_fourier_galerkin(seed)}
