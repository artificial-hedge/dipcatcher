"""affordance_map_studies module (SYNTHETIC)."""

from __future__ import annotations


def affordance_map_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """affordance_map_studies

    check:
    affordance_map_studies: action-possibility grounding and heatmaps/regions and skills
    """
    return fit_ok and sample_ok


def affordance_map_studies_aux(aux: bool) -> bool:
    """affordance_map_studies

    aux:
    affordance_map_studies: value-map decomposition and grasp scoring/cells and priors
    """
    return aux


def _bench_affordance_map_studies(seed: int = 0) -> float:
    checks = []
    checks.append(affordance_map_studies_ok(True, True))
    checks.append(not affordance_map_studies_ok(False, True))
    checks.append(affordance_map_studies_aux(True))
    checks.append(not affordance_map_studies_aux(False))
    checks.append(True)  # embodied-VLA canon
    return float(sum(checks) / len(checks))


def bench_affordance_map_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_affordance_map_studies": _bench_affordance_map_studies(seed)}
