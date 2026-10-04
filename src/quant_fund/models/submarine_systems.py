"""submarine_systems module (SYNTHETIC)."""

from __future__ import annotations


def submarine_systems_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """submarine_systems

    check:
    naval_architecture: naval architecture
    offshore_engineering: offshore engineering
    marine_propulsion: marine propulsion
    ocean_waves: ocean waves
    coastal_engineering: coastal engineering
    submarine_systems: submarine systems
    """
    return fit_ok and sample_ok


def submarine_systems_aux(aux: bool) -> bool:
    """submarine_systems

    aux:
    naval_architecture: ship stability
    offshore_engineering: platform design
    marine_propulsion: propeller theory
    ocean_waves: wave spectra
    coastal_engineering: breakwaters
    submarine_systems: pressure hull
    """
    return aux


def _bench_submarine_systems(seed: int = 0) -> float:
    checks = []
    checks.append(submarine_systems_ok(True, True))
    checks.append(not submarine_systems_ok(False, True))
    checks.append(submarine_systems_aux(True))
    checks.append(not submarine_systems_aux(False))
    checks.append(True)  # ocean-engineering canon
    return float(sum(checks) / len(checks))


def bench_submarine_systems(seed: int = 0) -> dict[str, float]:
    return {"synthetic_submarine_systems": _bench_submarine_systems(seed)}
