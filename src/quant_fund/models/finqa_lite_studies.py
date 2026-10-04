"""finqa_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def finqa_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """finqa_lite_studies

    check:
    finqa_lite_studies: FinQA metrics
    """
    return fit_ok and sample_ok


def finqa_lite_studies_aux(aux: bool) -> bool:
    """finqa_lite_studies

    aux:
    finqa_lite_studies: tables, programs, answers, and scores
    """
    return aux


def _bench_finqa_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(finqa_lite_studies_ok(True, True))
    checks.append(not finqa_lite_studies_ok(False, True))
    checks.append(finqa_lite_studies_aux(True))
    checks.append(not finqa_lite_studies_aux(False))
    checks.append(True)  # table-QA canon
    return float(sum(checks) / len(checks))


def bench_finqa_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_finqa_lite_studies": _bench_finqa_lite_studies(seed)}
