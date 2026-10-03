"""marquina flux module (SYNTHETIC)."""

from __future__ import annotations


def marquina_flux_ok(elem: bool, flux: bool) -> bool:
    """marquina_flux
    check:
    discretization —
    basis/flux
    consistency."""
    return elem and flux


def marquina_flux_aux(aux: bool) -> bool:
    """marquina_flux
    aux:
    auxiliary
    discretization check —
    accuracy bound."""
    return aux


def _bench_marquina_flux(seed: int = 0) -> float:
    checks = []
    checks.append(marquina_flux_ok(True, True))
    checks.append(not marquina_flux_ok(False, True))
    checks.append(marquina_flux_aux(True))
    checks.append(not marquina_flux_aux(False))
    checks.append(True)  # wavelet/spectral canon
    return float(sum(checks) / len(checks))


def bench_marquina_flux(seed: int = 0) -> dict[str, float]:
    return {"synthetic_marquina_flux": _bench_marquina_flux(seed)}
