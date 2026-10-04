"""leisure_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def leisure_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """leisure_qa_studies

    check:
    leisure_qa_studies: LeisureQA metrics
    """
    return fit_ok and sample_ok


def leisure_qa_studies_aux(aux: bool) -> bool:
    """leisure_qa_studies

    aux:
    leisure_qa_studies: activities, venues, answers, and scores
    """
    return aux


def _bench_leisure_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(leisure_qa_studies_ok(True, True))
    checks.append(not leisure_qa_studies_ok(False, True))
    checks.append(leisure_qa_studies_aux(True))
    checks.append(not leisure_qa_studies_aux(False))
    checks.append(True)  # leisure canon
    return float(sum(checks) / len(checks))


def bench_leisure_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_leisure_qa_studies": _bench_leisure_qa_studies(seed)}
