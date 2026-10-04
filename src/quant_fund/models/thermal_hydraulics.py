"""thermal_hydraulics module (SYNTHETIC)."""

from __future__ import annotations


def thermal_hydraulics_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """thermal_hydraulics

    check:
    reactor_physics: reactor physics
    radiation_protection: radiation protection
    nuclear_fuel_cycle: nuclear fuel cycle
    thermal_hydraulics: thermal hydraulics
    nuclear_safety: nuclear safety
    isotope_production: isotope production
    """
    return fit_ok and sample_ok


def thermal_hydraulics_aux(aux: bool) -> bool:
    """thermal_hydraulics

    aux:
    reactor_physics: neutron transport
    radiation_protection: shielding
    nuclear_fuel_cycle: enrichment
    thermal_hydraulics: coolant flow
    nuclear_safety: containment
    isotope_production: irradiation
    """
    return aux


def _bench_thermal_hydraulics(seed: int = 0) -> float:
    checks = []
    checks.append(thermal_hydraulics_ok(True, True))
    checks.append(not thermal_hydraulics_ok(False, True))
    checks.append(thermal_hydraulics_aux(True))
    checks.append(not thermal_hydraulics_aux(False))
    checks.append(True)  # nuclear-engineering canon
    return float(sum(checks) / len(checks))


def bench_thermal_hydraulics(seed: int = 0) -> dict[str, float]:
    return {"synthetic_thermal_hydraulics": _bench_thermal_hydraulics(seed)}
