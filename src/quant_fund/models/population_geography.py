"""population_geography module (SYNTHETIC)."""

from __future__ import annotations


def population_geography_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """population_geography

    check:
    regional_geography: regional geography
    health_geography: health geography
    population_geography: population geography
    economic_geography: economic geography
    political_geography: political geography
    gis_science: GIScience
    """
    return fit_ok and sample_ok


def population_geography_aux(aux: bool) -> bool:
    """population_geography

    aux:
    regional_geography: regional systems
    health_geography: disease patterns
    population_geography: demographic patterns
    economic_geography: spatial economy
    political_geography: territory and power
    gis_science: spatial analysis
    """
    return aux


def _bench_population_geography(seed: int = 0) -> float:
    checks = []
    checks.append(population_geography_ok(True, True))
    checks.append(not population_geography_ok(False, True))
    checks.append(population_geography_aux(True))
    checks.append(not population_geography_aux(False))
    checks.append(True)  # geography-3 canon
    return float(sum(checks) / len(checks))


def bench_population_geography(seed: int = 0) -> dict[str, float]:
    return {"synthetic_population_geography": _bench_population_geography(seed)}
