"""finch_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def finch_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """finch_qa_studies

    check:
    finch_qa_studies: FinchQA metrics
    """
    return fit_ok and sample_ok


def finch_qa_studies_aux(aux: bool) -> bool:
    """finch_qa_studies

    aux:
    finch_qa_studies: finches, meadows, answers, and scores
    """
    return aux


def _bench_finch_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(finch_qa_studies_ok(True, True))
    checks.append(not finch_qa_studies_ok(False, True))
    checks.append(finch_qa_studies_aux(True))
    checks.append(not finch_qa_studies_aux(False))
    checks.append(True)  # songbird canon
    return float(sum(checks) / len(checks))


def bench_finch_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_finch_qa_studies": _bench_finch_qa_studies(seed)}
