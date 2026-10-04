"""regional_science module (SYNTHETIC)."""

from __future__ import annotations


def regional_science_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """regional_science

    check:
    geography_2: geography
    regional_science: regional science
    demography_2: demography
    urbanization: urbanization
    land_use: land use
    gis_science_2: GIScience
    """
    return fit_ok and sample_ok


def regional_science_aux(aux: bool) -> bool:
    """regional_science

    aux:
    geography_2: places and spaces
    regional_science: regions and agglomeration
    demography_2: populations and cohorts
    urbanization: cities and growth
    land_use: parcels and zoning
    gis_science_2: rasters and vectors
    """
    return aux


def _bench_regional_science(seed: int = 0) -> float:
    checks = []
    checks.append(regional_science_ok(True, True))
    checks.append(not regional_science_ok(False, True))
    checks.append(regional_science_aux(True))
    checks.append(not regional_science_aux(False))
    checks.append(True)  # geography canon
    return float(sum(checks) / len(checks))


def bench_regional_science(seed: int = 0) -> dict[str, float]:
    return {"synthetic_regional_science": _bench_regional_science(seed)}
