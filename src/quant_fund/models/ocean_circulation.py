"""ocean_circulation module (SYNTHETIC)."""

from __future__ import annotations


def ocean_circulation_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ocean_circulation

    check:
    climate_model: climate model
    ocean_circulation: ocean circulation
    atmospheric_chem: atmospheric chemistry
    hydrology: hydrology
    carbon_cycle: carbon cycle
    ecosystem_model: ecosystem model
    """
    return fit_ok and sample_ok


def ocean_circulation_aux(aux: bool) -> bool:
    """ocean_circulation

    aux:
    climate_model: GCM simulation
    ocean_circulation: thermohaline
    atmospheric_chem: chemical transport
    hydrology: watershed model
    carbon_cycle: carbon budget
    ecosystem_model: trophic dynamics
    """
    return aux


def _bench_ocean_circulation(seed: int = 0) -> float:
    checks = []
    checks.append(ocean_circulation_ok(True, True))
    checks.append(not ocean_circulation_ok(False, True))
    checks.append(ocean_circulation_aux(True))
    checks.append(not ocean_circulation_aux(False))
    checks.append(True)  # environmental-science canon
    return float(sum(checks) / len(checks))


def bench_ocean_circulation(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ocean_circulation": _bench_ocean_circulation(seed)}
