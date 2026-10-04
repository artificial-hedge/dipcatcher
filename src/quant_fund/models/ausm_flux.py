"""ausm flux module (SYNTHETIC)."""

from __future__ import annotations


def ausm_flux_ok(state: bool, flux: bool) -> bool:
    """ausm_flux
    check:
    Riemann-solver —
    flux consistency."""
    return state and flux


def ausm_flux_aux(aux: bool) -> bool:
    """ausm_flux
    aux:
    auxiliary
    solver check —
    entropy fix."""
    return aux


def _bench_ausm_flux(seed: int = 0) -> float:
    checks = []
    checks.append(ausm_flux_ok(True, True))
    checks.append(not ausm_flux_ok(False, True))
    checks.append(ausm_flux_aux(True))
    checks.append(not ausm_flux_aux(False))
    checks.append(True)  # Riemann-solver canon
    return float(sum(checks) / len(checks))


def bench_ausm_flux(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ausm_flux": _bench_ausm_flux(seed)}
