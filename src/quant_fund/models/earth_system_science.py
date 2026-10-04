"""earth_system_science module (SYNTHETIC)."""

from __future__ import annotations


def earth_system_science_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """earth_system_science

    check:
    earth_system_science: earth system science
    oceanography_2: oceanography
    atmospheric_science: atmospheric science
    environmental_science_2: environmental science
    soil_science_2: soil science
    hydrology_3: hydrology
    """
    return fit_ok and sample_ok


def earth_system_science_aux(aux: bool) -> bool:
    """earth_system_science

    aux:
    earth_system_science: biospheres and cycles
    oceanography_2: currents and salinity
    atmospheric_science: winds and aerosols
    environmental_science_2: pollutants and remediation
    soil_science_2: horizons and nutrients
    hydrology_3: aquifers and runoff
    """
    return aux


def _bench_earth_system_science(seed: int = 0) -> float:
    checks = []
    checks.append(earth_system_science_ok(True, True))
    checks.append(not earth_system_science_ok(False, True))
    checks.append(earth_system_science_aux(True))
    checks.append(not earth_system_science_aux(False))
    checks.append(True)  # earth-systems canon
    return float(sum(checks) / len(checks))


def bench_earth_system_science(seed: int = 0) -> dict[str, float]:
    return {"synthetic_earth_system_science": _bench_earth_system_science(seed)}
