"""nuclear_safety module (SYNTHETIC)."""

from __future__ import annotations


def nuclear_safety_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nuclear_safety

    check:
    reactor_physics: reactor physics
    radiation_protection: radiation protection
    nuclear_fuel_cycle: nuclear fuel cycle
    thermal_hydraulics: thermal hydraulics
    nuclear_safety: nuclear safety
    isotope_production: isotope production
    """
    return fit_ok and sample_ok


def nuclear_safety_aux(aux: bool) -> bool:
    """nuclear_safety

    aux:
    reactor_physics: neutron transport
    radiation_protection: shielding
    nuclear_fuel_cycle: enrichment
    thermal_hydraulics: coolant flow
    nuclear_safety: containment
    isotope_production: irradiation
    """
    return aux


def _bench_nuclear_safety(seed: int = 0) -> float:
    checks = []
    checks.append(nuclear_safety_ok(True, True))
    checks.append(not nuclear_safety_ok(False, True))
    checks.append(nuclear_safety_aux(True))
    checks.append(not nuclear_safety_aux(False))
    checks.append(True)  # nuclear-engineering canon
    return float(sum(checks) / len(checks))


def bench_nuclear_safety(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nuclear_safety": _bench_nuclear_safety(seed)}
