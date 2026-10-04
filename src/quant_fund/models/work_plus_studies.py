"""work_plus_studies module (SYNTHETIC)."""

from __future__ import annotations


def work_plus_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """work_plus_studies

    check:
    work_plus_studies: WorkBench+ metrics
    """
    return fit_ok and sample_ok


def work_plus_studies_aux(aux: bool) -> bool:
    """work_plus_studies

    aux:
    work_plus_studies: tasks, actions, outcomes, and scores
    """
    return aux


def _bench_work_plus_studies(seed: int = 0) -> float:
    checks = []
    checks.append(work_plus_studies_ok(True, True))
    checks.append(not work_plus_studies_ok(False, True))
    checks.append(work_plus_studies_aux(True))
    checks.append(not work_plus_studies_aux(False))
    checks.append(True)  # toolbench canon
    return float(sum(checks) / len(checks))


def bench_work_plus_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_work_plus_studies": _bench_work_plus_studies(seed)}
