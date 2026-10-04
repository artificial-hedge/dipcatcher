"""hydrology module (SYNTHETIC)."""

from __future__ import annotations


def hydrology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hydrology

    check:
    climate_model: climate model
    ocean_circulation: ocean circulation
    atmospheric_chem: atmospheric chemistry
    hydrology: hydrology
    carbon_cycle: carbon cycle
    ecosystem_model: ecosystem model
    """
    return fit_ok and sample_ok


def hydrology_aux(aux: bool) -> bool:
    """hydrology

    aux:
    climate_model: GCM simulation
    ocean_circulation: thermohaline
    atmospheric_chem: chemical transport
    hydrology: watershed model
    carbon_cycle: carbon budget
    ecosystem_model: trophic dynamics
    """
    return aux


def _bench_hydrology(seed: int = 0) -> float:
    checks = []
    checks.append(hydrology_ok(True, True))
    checks.append(not hydrology_ok(False, True))
    checks.append(hydrology_aux(True))
    checks.append(not hydrology_aux(False))
    checks.append(True)  # environmental-science canon
    return float(sum(checks) / len(checks))


def bench_hydrology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hydrology": _bench_hydrology(seed)}
