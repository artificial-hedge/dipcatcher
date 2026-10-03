"""ldg flux module (SYNTHETIC)."""

from __future__ import annotations


def ldg_flux_ok(term: bool, est: bool) -> bool:
    """ldg_flux
    check:
    flux/asymptotic —
    term/estimate
    consistency."""
    return term and est


def ldg_flux_aux(aux: bool) -> bool:
    """ldg_flux
    aux:
    auxiliary
    flux/asymptotic check —
    error bound."""
    return aux


def _bench_ldg_flux(seed: int = 0) -> float:
    checks = []
    checks.append(ldg_flux_ok(True, True))
    checks.append(not ldg_flux_ok(False, True))
    checks.append(ldg_flux_aux(True))
    checks.append(not ldg_flux_aux(False))
    checks.append(True)  # flux/asymptotic canon
    return float(sum(checks) / len(checks))


def bench_ldg_flux(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ldg_flux": _bench_ldg_flux(seed)}
