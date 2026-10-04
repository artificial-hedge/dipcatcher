"""land_use module (SYNTHETIC)."""

from __future__ import annotations


def land_use_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """land_use

    check:
    geography_2: geography
    regional_science: regional science
    demography_2: demography
    urbanization: urbanization
    land_use: land use
    gis_science_2: GIScience
    """
    return fit_ok and sample_ok


def land_use_aux(aux: bool) -> bool:
    """land_use

    aux:
    geography_2: places and spaces
    regional_science: regions and agglomeration
    demography_2: populations and cohorts
    urbanization: cities and growth
    land_use: parcels and zoning
    gis_science_2: rasters and vectors
    """
    return aux


def _bench_land_use(seed: int = 0) -> float:
    checks = []
    checks.append(land_use_ok(True, True))
    checks.append(not land_use_ok(False, True))
    checks.append(land_use_aux(True))
    checks.append(not land_use_aux(False))
    checks.append(True)  # geography canon
    return float(sum(checks) / len(checks))


def bench_land_use(seed: int = 0) -> dict[str, float]:
    return {"synthetic_land_use": _bench_land_use(seed)}
