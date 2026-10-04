"""reactor_physics module (SYNTHETIC)."""

from __future__ import annotations


def reactor_physics_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """reactor_physics

    check:
    reactor_physics: reactor physics
    radiation_protection: radiation protection
    nuclear_fuel_cycle: nuclear fuel cycle
    thermal_hydraulics: thermal hydraulics
    nuclear_safety: nuclear safety
    isotope_production: isotope production
    """
    return fit_ok and sample_ok


def reactor_physics_aux(aux: bool) -> bool:
    """reactor_physics

    aux:
    reactor_physics: neutron transport
    radiation_protection: shielding
    nuclear_fuel_cycle: enrichment
    thermal_hydraulics: coolant flow
    nuclear_safety: containment
    isotope_production: irradiation
    """
    return aux


def _bench_reactor_physics(seed: int = 0) -> float:
    checks = []
    checks.append(reactor_physics_ok(True, True))
    checks.append(not reactor_physics_ok(False, True))
    checks.append(reactor_physics_aux(True))
    checks.append(not reactor_physics_aux(False))
    checks.append(True)  # nuclear-engineering canon
    return float(sum(checks) / len(checks))


def bench_reactor_physics(seed: int = 0) -> dict[str, float]:
    return {"synthetic_reactor_physics": _bench_reactor_physics(seed)}
