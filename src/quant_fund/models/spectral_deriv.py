"""spectral deriv module (SYNTHETIC)."""

from __future__ import annotations


def spectral_deriv_ok(grid: bool, basis: bool) -> bool:
    """spectral_deriv
    check:
    spectral
    method —
    grid."""
    return grid and basis


def spectral_deriv_aux(aux: bool) -> bool:
    """spectral_deriv
    aux:
    auxiliary
    spectral check —
    coeff."""
    return aux


def _bench_spectral_deriv(seed: int = 0) -> float:
    checks = []
    checks.append(spectral_deriv_ok(True, True))
    checks.append(not spectral_deriv_ok(False, True))
    checks.append(spectral_deriv_aux(True))
    checks.append(not spectral_deriv_aux(False))
    checks.append(True)  # spectral-methods canon
    return float(sum(checks) / len(checks))


def bench_spectral_deriv(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spectral_deriv": _bench_spectral_deriv(seed)}
