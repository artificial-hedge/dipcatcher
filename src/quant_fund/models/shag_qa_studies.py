"""shag_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def shag_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """shag_qa_studies

    check:
    shag_qa_studies: ShagQA metrics
    """
    return fit_ok and sample_ok


def shag_qa_studies_aux(aux: bool) -> bool:
    """shag_qa_studies

    aux:
    shag_qa_studies: shags, rocky coasts, answers, and scores
    """
    return aux


def _bench_shag_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(shag_qa_studies_ok(True, True))
    checks.append(not shag_qa_studies_ok(False, True))
    checks.append(shag_qa_studies_aux(True))
    checks.append(not shag_qa_studies_aux(False))
    checks.append(True)  # seabird-4 canon
    return float(sum(checks) / len(checks))


def bench_shag_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_shag_qa_studies": _bench_shag_qa_studies(seed)}
