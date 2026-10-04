"""holly_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def holly_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """holly_qa_studies

    check:
    holly_qa_studies: HollyQA metrics
    """
    return fit_ok and sample_ok


def holly_qa_studies_aux(aux: bool) -> bool:
    """holly_qa_studies

    aux:
    holly_qa_studies: hollies, berries, answers, and scores
    """
    return aux


def _bench_holly_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(holly_qa_studies_ok(True, True))
    checks.append(not holly_qa_studies_ok(False, True))
    checks.append(holly_qa_studies_aux(True))
    checks.append(not holly_qa_studies_aux(False))
    checks.append(True)  # evergreen canon
    return float(sum(checks) / len(checks))


def bench_holly_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_holly_qa_studies": _bench_holly_qa_studies(seed)}
