"""babilong_studies module (SYNTHETIC)."""

from __future__ import annotations


def babilong_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """babilong_studies

    check:
    babilong_studies: BABILong scattered-fact reasoning accuracy metrics
    """
    return fit_ok and sample_ok


def babilong_studies_aux(aux: bool) -> bool:
    """babilong_studies

    aux:
    babilong_studies: facts, distractors, questions, and scores
    """
    return aux


def _bench_babilong_studies(seed: int = 0) -> float:
    checks = []
    checks.append(babilong_studies_ok(True, True))
    checks.append(not babilong_studies_ok(False, True))
    checks.append(babilong_studies_aux(True))
    checks.append(not babilong_studies_aux(False))
    checks.append(True)  # long-context-eval canon
    return float(sum(checks) / len(checks))


def bench_babilong_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_babilong_studies": _bench_babilong_studies(seed)}
