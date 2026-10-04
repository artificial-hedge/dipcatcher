"""grailqa_studies module (SYNTHETIC)."""

from __future__ import annotations


def grailqa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """grailqa_studies

    check:
    grailqa_studies: GrailQA metrics
    """
    return fit_ok and sample_ok


def grailqa_studies_aux(aux: bool) -> bool:
    """grailqa_studies

    aux:
    grailqa_studies: questions, logical forms, answers, and scores
    """
    return aux


def _bench_grailqa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(grailqa_studies_ok(True, True))
    checks.append(not grailqa_studies_ok(False, True))
    checks.append(grailqa_studies_aux(True))
    checks.append(not grailqa_studies_aux(False))
    checks.append(True)  # KG-QA canon
    return float(sum(checks) / len(checks))


def bench_grailqa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_grailqa_studies": _bench_grailqa_studies(seed)}
