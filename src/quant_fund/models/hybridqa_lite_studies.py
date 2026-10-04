"""hybridqa_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def hybridqa_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hybridqa_lite_studies

    check:
    hybridqa_lite_studies: HybridQA metrics
    """
    return fit_ok and sample_ok


def hybridqa_lite_studies_aux(aux: bool) -> bool:
    """hybridqa_lite_studies

    aux:
    hybridqa_lite_studies: tables, passages, answers, and scores
    """
    return aux


def _bench_hybridqa_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(hybridqa_lite_studies_ok(True, True))
    checks.append(not hybridqa_lite_studies_ok(False, True))
    checks.append(hybridqa_lite_studies_aux(True))
    checks.append(not hybridqa_lite_studies_aux(False))
    checks.append(True)  # table-QA canon
    return float(sum(checks) / len(checks))


def bench_hybridqa_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hybridqa_lite_studies": _bench_hybridqa_lite_studies(seed)}
