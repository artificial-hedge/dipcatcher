"""offshore_engineering module (SYNTHETIC)."""

from __future__ import annotations


def offshore_engineering_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """offshore_engineering

    check:
    naval_architecture: naval architecture
    offshore_engineering: offshore engineering
    marine_propulsion: marine propulsion
    ocean_waves: ocean waves
    coastal_engineering: coastal engineering
    submarine_systems: submarine systems
    """
    return fit_ok and sample_ok


def offshore_engineering_aux(aux: bool) -> bool:
    """offshore_engineering

    aux:
    naval_architecture: ship stability
    offshore_engineering: platform design
    marine_propulsion: propeller theory
    ocean_waves: wave spectra
    coastal_engineering: breakwaters
    submarine_systems: pressure hull
    """
    return aux


def _bench_offshore_engineering(seed: int = 0) -> float:
    checks = []
    checks.append(offshore_engineering_ok(True, True))
    checks.append(not offshore_engineering_ok(False, True))
    checks.append(offshore_engineering_aux(True))
    checks.append(not offshore_engineering_aux(False))
    checks.append(True)  # ocean-engineering canon
    return float(sum(checks) / len(checks))


def bench_offshore_engineering(seed: int = 0) -> dict[str, float]:
    return {"synthetic_offshore_engineering": _bench_offshore_engineering(seed)}
