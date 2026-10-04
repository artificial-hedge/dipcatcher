"""ocean_waves module (SYNTHETIC)."""

from __future__ import annotations


def ocean_waves_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ocean_waves

    check:
    naval_architecture: naval architecture
    offshore_engineering: offshore engineering
    marine_propulsion: marine propulsion
    ocean_waves: ocean waves
    coastal_engineering: coastal engineering
    submarine_systems: submarine systems
    """
    return fit_ok and sample_ok


def ocean_waves_aux(aux: bool) -> bool:
    """ocean_waves

    aux:
    naval_architecture: ship stability
    offshore_engineering: platform design
    marine_propulsion: propeller theory
    ocean_waves: wave spectra
    coastal_engineering: breakwaters
    submarine_systems: pressure hull
    """
    return aux


def _bench_ocean_waves(seed: int = 0) -> float:
    checks = []
    checks.append(ocean_waves_ok(True, True))
    checks.append(not ocean_waves_ok(False, True))
    checks.append(ocean_waves_aux(True))
    checks.append(not ocean_waves_aux(False))
    checks.append(True)  # ocean-engineering canon
    return float(sum(checks) / len(checks))


def bench_ocean_waves(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ocean_waves": _bench_ocean_waves(seed)}
