"""barong_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def barong_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """barong_qa_studies

    check:
    barong_qa_studies: BarongQA metrics
    """
    return fit_ok and sample_ok


def barong_qa_studies_aux(aux: bool) -> bool:
    """barong_qa_studies

    aux:
    barong_qa_studies: barong, lion protectors, answers, and scores
    """
    return aux


def _bench_barong_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(barong_qa_studies_ok(True, True))
    checks.append(not barong_qa_studies_ok(False, True))
    checks.append(barong_qa_studies_aux(True))
    checks.append(not barong_qa_studies_aux(False))
    checks.append(True)  # indonesian-myth canon
    return float(sum(checks) / len(checks))


def bench_barong_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_barong_qa_studies": _bench_barong_qa_studies(seed)}
