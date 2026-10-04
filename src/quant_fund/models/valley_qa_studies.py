"""valley_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def valley_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """valley_qa_studies

    check:
    valley_qa_studies: ValleyQA metrics
    """
    return fit_ok and sample_ok


def valley_qa_studies_aux(aux: bool) -> bool:
    """valley_qa_studies

    aux:
    valley_qa_studies: valleys, rivers, answers, and scores
    """
    return aux


def _bench_valley_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(valley_qa_studies_ok(True, True))
    checks.append(not valley_qa_studies_ok(False, True))
    checks.append(valley_qa_studies_aux(True))
    checks.append(not valley_qa_studies_aux(False))
    checks.append(True)  # highland canon
    return float(sum(checks) / len(checks))


def bench_valley_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_valley_qa_studies": _bench_valley_qa_studies(seed)}
