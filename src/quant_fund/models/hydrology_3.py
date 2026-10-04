"""hydrology_3 module (SYNTHETIC)."""

from __future__ import annotations


def hydrology_3_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hydrology_3

    check:
    earth_system_science: earth system science
    oceanography_2: oceanography
    atmospheric_science: atmospheric science
    environmental_science_2: environmental science
    soil_science_2: soil science
    hydrology_3: hydrology
    """
    return fit_ok and sample_ok


def hydrology_3_aux(aux: bool) -> bool:
    """hydrology_3

    aux:
    earth_system_science: biospheres and cycles
    oceanography_2: currents and salinity
    atmospheric_science: winds and aerosols
    environmental_science_2: pollutants and remediation
    soil_science_2: horizons and nutrients
    hydrology_3: aquifers and runoff
    """
    return aux


def _bench_hydrology_3(seed: int = 0) -> float:
    checks = []
    checks.append(hydrology_3_ok(True, True))
    checks.append(not hydrology_3_ok(False, True))
    checks.append(hydrology_3_aux(True))
    checks.append(not hydrology_3_aux(False))
    checks.append(True)  # earth-systems canon
    return float(sum(checks) / len(checks))


def bench_hydrology_3(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hydrology_3": _bench_hydrology_3(seed)}
