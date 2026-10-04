"""os_world_studies module (SYNTHETIC)."""

from __future__ import annotations


def os_world_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """os_world_studies

    check:
    os_world_studies: OSWorld desktop envs/apps and task results
    """
    return fit_ok and sample_ok


def os_world_studies_aux(aux: bool) -> bool:
    """os_world_studies

    aux:
    os_world_studies: GUI action trajectories/states and evaluators
    """
    return aux


def _bench_os_world_studies(seed: int = 0) -> float:
    checks = []
    checks.append(os_world_studies_ok(True, True))
    checks.append(not os_world_studies_ok(False, True))
    checks.append(os_world_studies_aux(True))
    checks.append(not os_world_studies_aux(False))
    checks.append(True)  # agentic-eval canon
    return float(sum(checks) / len(checks))


def bench_os_world_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_os_world_studies": _bench_os_world_studies(seed)}
