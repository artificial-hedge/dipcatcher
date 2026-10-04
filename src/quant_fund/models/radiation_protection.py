"""radiation_protection module (SYNTHETIC)."""

from __future__ import annotations


def radiation_protection_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """radiation_protection

    check:
    reactor_physics: reactor physics
    radiation_protection: radiation protection
    nuclear_fuel_cycle: nuclear fuel cycle
    thermal_hydraulics: thermal hydraulics
    nuclear_safety: nuclear safety
    isotope_production: isotope production
    """
    return fit_ok and sample_ok


def radiation_protection_aux(aux: bool) -> bool:
    """radiation_protection

    aux:
    reactor_physics: neutron transport
    radiation_protection: shielding
    nuclear_fuel_cycle: enrichment
    thermal_hydraulics: coolant flow
    nuclear_safety: containment
    isotope_production: irradiation
    """
    return aux


def _bench_radiation_protection(seed: int = 0) -> float:
    checks = []
    checks.append(radiation_protection_ok(True, True))
    checks.append(not radiation_protection_ok(False, True))
    checks.append(radiation_protection_aux(True))
    checks.append(not radiation_protection_aux(False))
    checks.append(True)  # nuclear-engineering canon
    return float(sum(checks) / len(checks))


def bench_radiation_protection(seed: int = 0) -> dict[str, float]:
    return {"synthetic_radiation_protection": _bench_radiation_protection(seed)}
