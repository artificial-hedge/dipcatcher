"""thrush_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def thrush_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """thrush_qa_studies

    check:
    thrush_qa_studies: ThrushQA metrics
    """
    return fit_ok and sample_ok


def thrush_qa_studies_aux(aux: bool) -> bool:
    """thrush_qa_studies

    aux:
    thrush_qa_studies: thrushes, gardens, answers, and scores
    """
    return aux


def _bench_thrush_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(thrush_qa_studies_ok(True, True))
    checks.append(not thrush_qa_studies_ok(False, True))
    checks.append(thrush_qa_studies_aux(True))
    checks.append(not thrush_qa_studies_aux(False))
    checks.append(True)  # songbird canon
    return float(sum(checks) / len(checks))


def bench_thrush_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_thrush_qa_studies": _bench_thrush_qa_studies(seed)}
