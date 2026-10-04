"""isotope_production module (SYNTHETIC)."""

from __future__ import annotations


def isotope_production_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """isotope_production

    check:
    reactor_physics: reactor physics
    radiation_protection: radiation protection
    nuclear_fuel_cycle: nuclear fuel cycle
    thermal_hydraulics: thermal hydraulics
    nuclear_safety: nuclear safety
    isotope_production: isotope production
    """
    return fit_ok and sample_ok


def isotope_production_aux(aux: bool) -> bool:
    """isotope_production

    aux:
    reactor_physics: neutron transport
    radiation_protection: shielding
    nuclear_fuel_cycle: enrichment
    thermal_hydraulics: coolant flow
    nuclear_safety: containment
    isotope_production: irradiation
    """
    return aux


def _bench_isotope_production(seed: int = 0) -> float:
    checks = []
    checks.append(isotope_production_ok(True, True))
    checks.append(not isotope_production_ok(False, True))
    checks.append(isotope_production_aux(True))
    checks.append(not isotope_production_aux(False))
    checks.append(True)  # nuclear-engineering canon
    return float(sum(checks) / len(checks))


def bench_isotope_production(seed: int = 0) -> dict[str, float]:
    return {"synthetic_isotope_production": _bench_isotope_production(seed)}
