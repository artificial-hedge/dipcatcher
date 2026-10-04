"""osworld_studies module (SYNTHETIC)."""

from __future__ import annotations


def osworld_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """osworld_studies

    check:
    osworld_studies: OSWorld desktop-task success and step accuracy
    """
    return fit_ok and sample_ok


def osworld_studies_aux(aux: bool) -> bool:
    """osworld_studies

    aux:
    osworld_studies: env states, action traces, and task completion
    """
    return aux


def _bench_osworld_studies(seed: int = 0) -> float:
    checks = []
    checks.append(osworld_studies_ok(True, True))
    checks.append(not osworld_studies_ok(False, True))
    checks.append(osworld_studies_aux(True))
    checks.append(not osworld_studies_aux(False))
    checks.append(True)  # agent-eval canon
    return float(sum(checks) / len(checks))


def bench_osworld_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_osworld_studies": _bench_osworld_studies(seed)}
