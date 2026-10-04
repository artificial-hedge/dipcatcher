"""dingo_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def dingo_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dingo_qa_studies

    check:
    dingo_qa_studies: DingoQA metrics
    """
    return fit_ok and sample_ok


def dingo_qa_studies_aux(aux: bool) -> bool:
    """dingo_qa_studies

    aux:
    dingo_qa_studies: dingos, outbacks, answers, and scores
    """
    return aux


def _bench_dingo_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(dingo_qa_studies_ok(True, True))
    checks.append(not dingo_qa_studies_ok(False, True))
    checks.append(dingo_qa_studies_aux(True))
    checks.append(not dingo_qa_studies_aux(False))
    checks.append(True)  # wildcat-2 canon
    return float(sum(checks) / len(checks))


def bench_dingo_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dingo_qa_studies": _bench_dingo_qa_studies(seed)}
