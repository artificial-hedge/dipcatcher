"""asqa_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def asqa_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """asqa_lite_studies

    check:
    asqa_lite_studies: ASQA metrics
    """
    return fit_ok and sample_ok


def asqa_lite_studies_aux(aux: bool) -> bool:
    """asqa_lite_studies

    aux:
    asqa_lite_studies: questions, passages, answers, and scores
    """
    return aux


def _bench_asqa_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(asqa_lite_studies_ok(True, True))
    checks.append(not asqa_lite_studies_ok(False, True))
    checks.append(asqa_lite_studies_aux(True))
    checks.append(not asqa_lite_studies_aux(False))
    checks.append(True)  # retrieval-eval canon
    return float(sum(checks) / len(checks))


def bench_asqa_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_asqa_lite_studies": _bench_asqa_lite_studies(seed)}
