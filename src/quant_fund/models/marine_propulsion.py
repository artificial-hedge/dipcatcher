"""marine_propulsion module (SYNTHETIC)."""

from __future__ import annotations


def marine_propulsion_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """marine_propulsion

    check:
    naval_architecture: naval architecture
    offshore_engineering: offshore engineering
    marine_propulsion: marine propulsion
    ocean_waves: ocean waves
    coastal_engineering: coastal engineering
    submarine_systems: submarine systems
    """
    return fit_ok and sample_ok


def marine_propulsion_aux(aux: bool) -> bool:
    """marine_propulsion

    aux:
    naval_architecture: ship stability
    offshore_engineering: platform design
    marine_propulsion: propeller theory
    ocean_waves: wave spectra
    coastal_engineering: breakwaters
    submarine_systems: pressure hull
    """
    return aux


def _bench_marine_propulsion(seed: int = 0) -> float:
    checks = []
    checks.append(marine_propulsion_ok(True, True))
    checks.append(not marine_propulsion_ok(False, True))
    checks.append(marine_propulsion_aux(True))
    checks.append(not marine_propulsion_aux(False))
    checks.append(True)  # ocean-engineering canon
    return float(sum(checks) / len(checks))


def bench_marine_propulsion(seed: int = 0) -> dict[str, float]:
    return {"synthetic_marine_propulsion": _bench_marine_propulsion(seed)}
