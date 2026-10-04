"""urbanization module (SYNTHETIC)."""

from __future__ import annotations


def urbanization_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """urbanization

    check:
    geography_2: geography
    regional_science: regional science
    demography_2: demography
    urbanization: urbanization
    land_use: land use
    gis_science_2: GIScience
    """
    return fit_ok and sample_ok


def urbanization_aux(aux: bool) -> bool:
    """urbanization

    aux:
    geography_2: places and spaces
    regional_science: regions and agglomeration
    demography_2: populations and cohorts
    urbanization: cities and growth
    land_use: parcels and zoning
    gis_science_2: rasters and vectors
    """
    return aux


def _bench_urbanization(seed: int = 0) -> float:
    checks = []
    checks.append(urbanization_ok(True, True))
    checks.append(not urbanization_ok(False, True))
    checks.append(urbanization_aux(True))
    checks.append(not urbanization_aux(False))
    checks.append(True)  # geography canon
    return float(sum(checks) / len(checks))


def bench_urbanization(seed: int = 0) -> dict[str, float]:
    return {"synthetic_urbanization": _bench_urbanization(seed)}
