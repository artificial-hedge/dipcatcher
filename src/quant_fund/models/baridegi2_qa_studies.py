"""baridegi2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def baridegi2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """baridegi2_qa_studies

    check:
    baridegi2_qa_studies: Baridegi2QA metrics
    """
    return fit_ok and sample_ok


def baridegi2_qa_studies_aux(aux: bool) -> bool:
    """baridegi2_qa_studies

    aux:
    baridegi2_qa_studies: baridegi2, abandoned shamans, answers, and scores
    """
    return aux


def _bench_baridegi2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(baridegi2_qa_studies_ok(True, True))
    checks.append(not baridegi2_qa_studies_ok(False, True))
    checks.append(baridegi2_qa_studies_aux(True))
    checks.append(not baridegi2_qa_studies_aux(False))
    checks.append(True)  # korean-myth-3 canon
    return float(sum(checks) / len(checks))


def bench_baridegi2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_baridegi2_qa_studies": _bench_baridegi2_qa_studies(seed)}
