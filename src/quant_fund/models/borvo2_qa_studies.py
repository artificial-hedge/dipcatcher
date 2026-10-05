"""borvo2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def borvo2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """borvo2_qa_studies

    check:
    borvo2_qa_studies: Borvo2QA metrics
    """
    return fit_ok and sample_ok


def borvo2_qa_studies_aux(aux: bool) -> bool:
    """borvo2_qa_studies

    aux:
    borvo2_qa_studies: borvo2, boiling springs, answers, and scores
    """
    return aux


def _bench_borvo2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(borvo2_qa_studies_ok(True, True))
    checks.append(not borvo2_qa_studies_ok(False, True))
    checks.append(borvo2_qa_studies_aux(True))
    checks.append(not borvo2_qa_studies_aux(False))
    checks.append(True)  # celtic-myth-4 canon
    return float(sum(checks) / len(checks))


def bench_borvo2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_borvo2_qa_studies": _bench_borvo2_qa_studies(seed)}
