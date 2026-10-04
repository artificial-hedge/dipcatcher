"""airtasks_studies module (SYNTHETIC)."""

from __future__ import annotations


def airtasks_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """airtasks_studies

    check:
    airtasks_studies: AirTasks flight metrics
    """
    return fit_ok and sample_ok


def airtasks_studies_aux(aux: bool) -> bool:
    """airtasks_studies

    aux:
    airtasks_studies: tasks, actions, states, and scores
    """
    return aux


def _bench_airtasks_studies(seed: int = 0) -> float:
    checks = []
    checks.append(airtasks_studies_ok(True, True))
    checks.append(not airtasks_studies_ok(False, True))
    checks.append(airtasks_studies_aux(True))
    checks.append(not airtasks_studies_aux(False))
    checks.append(True)  # web-agent canon
    return float(sum(checks) / len(checks))


def bench_airtasks_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_airtasks_studies": _bench_airtasks_studies(seed)}
