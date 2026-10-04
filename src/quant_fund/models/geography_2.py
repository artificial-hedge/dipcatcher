"""geography_2 module (SYNTHETIC)."""

from __future__ import annotations


def geography_2_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """geography_2

    check:
    geography_2: geography
    regional_science: regional science
    demography_2: demography
    urbanization: urbanization
    land_use: land use
    gis_science_2: GIScience
    """
    return fit_ok and sample_ok


def geography_2_aux(aux: bool) -> bool:
    """geography_2

    aux:
    geography_2: places and spaces
    regional_science: regions and agglomeration
    demography_2: populations and cohorts
    urbanization: cities and growth
    land_use: parcels and zoning
    gis_science_2: rasters and vectors
    """
    return aux


def _bench_geography_2(seed: int = 0) -> float:
    checks = []
    checks.append(geography_2_ok(True, True))
    checks.append(not geography_2_ok(False, True))
    checks.append(geography_2_aux(True))
    checks.append(not geography_2_aux(False))
    checks.append(True)  # geography canon
    return float(sum(checks) / len(checks))


def bench_geography_2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_geography_2": _bench_geography_2(seed)}
