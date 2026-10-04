"""needletail_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def needletail_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """needletail_qa_studies

    check:
    needletail_qa_studies: NeedletailQA metrics
    """
    return fit_ok and sample_ok


def needletail_qa_studies_aux(aux: bool) -> bool:
    """needletail_qa_studies

    aux:
    needletail_qa_studies: needletails, ridges, answers, and scores
    """
    return aux


def _bench_needletail_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(needletail_qa_studies_ok(True, True))
    checks.append(not needletail_qa_studies_ok(False, True))
    checks.append(needletail_qa_studies_aux(True))
    checks.append(not needletail_qa_studies_aux(False))
    checks.append(True)  # aerialist canon
    return float(sum(checks) / len(checks))


def bench_needletail_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_needletail_qa_studies": _bench_needletail_qa_studies(seed)}
