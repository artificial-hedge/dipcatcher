"""perch_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def perch_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """perch_qa_studies

    check:
    perch_qa_studies: PerchQA metrics
    """
    return fit_ok and sample_ok


def perch_qa_studies_aux(aux: bool) -> bool:
    """perch_qa_studies

    aux:
    perch_qa_studies: perches, lake schools, answers, and scores
    """
    return aux


def _bench_perch_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(perch_qa_studies_ok(True, True))
    checks.append(not perch_qa_studies_ok(False, True))
    checks.append(perch_qa_studies_aux(True))
    checks.append(not perch_qa_studies_aux(False))
    checks.append(True)  # freshwater-fish canon
    return float(sum(checks) / len(checks))


def bench_perch_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_perch_qa_studies": _bench_perch_qa_studies(seed)}
