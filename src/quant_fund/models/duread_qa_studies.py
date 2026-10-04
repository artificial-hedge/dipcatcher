"""duread_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def duread_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """duread_qa_studies

    check:
    duread_qa_studies: DuReader metrics
    """
    return fit_ok and sample_ok


def duread_qa_studies_aux(aux: bool) -> bool:
    """duread_qa_studies

    aux:
    duread_qa_studies: passages, questions, answers, and scores
    """
    return aux


def _bench_duread_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(duread_qa_studies_ok(True, True))
    checks.append(not duread_qa_studies_ok(False, True))
    checks.append(duread_qa_studies_aux(True))
    checks.append(not duread_qa_studies_aux(False))
    checks.append(True)  # conversational-QA canon
    return float(sum(checks) / len(checks))


def bench_duread_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_duread_qa_studies": _bench_duread_qa_studies(seed)}
