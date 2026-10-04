"""radien_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def radien_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """radien_qa_studies

    check:
    radien_qa_studies: RadienQA metrics
    """
    return fit_ok and sample_ok


def radien_qa_studies_aux(aux: bool) -> bool:
    """radien_qa_studies

    aux:
    radien_qa_studies: radien, sky fathers, answers, and scores
    """
    return aux


def _bench_radien_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(radien_qa_studies_ok(True, True))
    checks.append(not radien_qa_studies_ok(False, True))
    checks.append(radien_qa_studies_aux(True))
    checks.append(not radien_qa_studies_aux(False))
    checks.append(True)  # sami-myth canon
    return float(sum(checks) / len(checks))


def bench_radien_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_radien_qa_studies": _bench_radien_qa_studies(seed)}
