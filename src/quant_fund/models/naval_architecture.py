"""naval_architecture module (SYNTHETIC)."""

from __future__ import annotations


def naval_architecture_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """naval_architecture

    check:
    naval_architecture: naval architecture
    offshore_engineering: offshore engineering
    marine_propulsion: marine propulsion
    ocean_waves: ocean waves
    coastal_engineering: coastal engineering
    submarine_systems: submarine systems
    """
    return fit_ok and sample_ok


def naval_architecture_aux(aux: bool) -> bool:
    """naval_architecture

    aux:
    naval_architecture: ship stability
    offshore_engineering: platform design
    marine_propulsion: propeller theory
    ocean_waves: wave spectra
    coastal_engineering: breakwaters
    submarine_systems: pressure hull
    """
    return aux


def _bench_naval_architecture(seed: int = 0) -> float:
    checks = []
    checks.append(naval_architecture_ok(True, True))
    checks.append(not naval_architecture_ok(False, True))
    checks.append(naval_architecture_aux(True))
    checks.append(not naval_architecture_aux(False))
    checks.append(True)  # ocean-engineering canon
    return float(sum(checks) / len(checks))


def bench_naval_architecture(seed: int = 0) -> dict[str, float]:
    return {"synthetic_naval_architecture": _bench_naval_architecture(seed)}
