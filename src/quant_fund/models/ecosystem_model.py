"""ecosystem_model module (SYNTHETIC)."""

from __future__ import annotations


def ecosystem_model_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ecosystem_model

    check:
    climate_model: climate model
    ocean_circulation: ocean circulation
    atmospheric_chem: atmospheric chemistry
    hydrology: hydrology
    carbon_cycle: carbon cycle
    ecosystem_model: ecosystem model
    """
    return fit_ok and sample_ok


def ecosystem_model_aux(aux: bool) -> bool:
    """ecosystem_model

    aux:
    climate_model: GCM simulation
    ocean_circulation: thermohaline
    atmospheric_chem: chemical transport
    hydrology: watershed model
    carbon_cycle: carbon budget
    ecosystem_model: trophic dynamics
    """
    return aux


def _bench_ecosystem_model(seed: int = 0) -> float:
    checks = []
    checks.append(ecosystem_model_ok(True, True))
    checks.append(not ecosystem_model_ok(False, True))
    checks.append(ecosystem_model_aux(True))
    checks.append(not ecosystem_model_aux(False))
    checks.append(True)  # environmental-science canon
    return float(sum(checks) / len(checks))


def bench_ecosystem_model(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ecosystem_model": _bench_ecosystem_model(seed)}
