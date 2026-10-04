"""butte_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def butte_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """butte_qa_studies

    check:
    butte_qa_studies: ButteQA metrics
    """
    return fit_ok and sample_ok


def butte_qa_studies_aux(aux: bool) -> bool:
    """butte_qa_studies

    aux:
    butte_qa_studies: buttes, mesas, answers, and scores
    """
    return aux


def _bench_butte_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(butte_qa_studies_ok(True, True))
    checks.append(not butte_qa_studies_ok(False, True))
    checks.append(butte_qa_studies_aux(True))
    checks.append(not butte_qa_studies_aux(False))
    checks.append(True)  # desert-2 canon
    return float(sum(checks) / len(checks))


def bench_butte_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_butte_qa_studies": _bench_butte_qa_studies(seed)}
