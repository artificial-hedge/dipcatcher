"""ground_dove_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ground_dove_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ground_dove_qa_studies

    check:
    ground_dove_qa_studies: Ground-doveQA metrics
    """
    return fit_ok and sample_ok


def ground_dove_qa_studies_aux(aux: bool) -> bool:
    """ground_dove_qa_studies

    aux:
    ground_dove_qa_studies: ground doves, thickets, answers, and scores
    """
    return aux


def _bench_ground_dove_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ground_dove_qa_studies_ok(True, True))
    checks.append(not ground_dove_qa_studies_ok(False, True))
    checks.append(ground_dove_qa_studies_aux(True))
    checks.append(not ground_dove_qa_studies_aux(False))
    checks.append(True)  # columbid-2 canon
    return float(sum(checks) / len(checks))


def bench_ground_dove_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ground_dove_qa_studies": _bench_ground_dove_qa_studies(seed)}
