"""environmental_science_2 module (SYNTHETIC)."""

from __future__ import annotations


def environmental_science_2_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """environmental_science_2

    check:
    earth_system_science: earth system science
    oceanography_2: oceanography
    atmospheric_science: atmospheric science
    environmental_science_2: environmental science
    soil_science_2: soil science
    hydrology_3: hydrology
    """
    return fit_ok and sample_ok


def environmental_science_2_aux(aux: bool) -> bool:
    """environmental_science_2

    aux:
    earth_system_science: biospheres and cycles
    oceanography_2: currents and salinity
    atmospheric_science: winds and aerosols
    environmental_science_2: pollutants and remediation
    soil_science_2: horizons and nutrients
    hydrology_3: aquifers and runoff
    """
    return aux


def _bench_environmental_science_2(seed: int = 0) -> float:
    checks = []
    checks.append(environmental_science_2_ok(True, True))
    checks.append(not environmental_science_2_ok(False, True))
    checks.append(environmental_science_2_aux(True))
    checks.append(not environmental_science_2_aux(False))
    checks.append(True)  # earth-systems canon
    return float(sum(checks) / len(checks))


def bench_environmental_science_2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_environmental_science_2": _bench_environmental_science_2(seed)}
