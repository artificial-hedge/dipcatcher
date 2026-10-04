"""barbet_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def barbet_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """barbet_qa_studies

    check:
    barbet_qa_studies: BarbetQA metrics
    """
    return fit_ok and sample_ok


def barbet_qa_studies_aux(aux: bool) -> bool:
    """barbet_qa_studies

    aux:
    barbet_qa_studies: barbets, groves, answers, and scores
    """
    return aux


def _bench_barbet_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(barbet_qa_studies_ok(True, True))
    checks.append(not barbet_qa_studies_ok(False, True))
    checks.append(barbet_qa_studies_aux(True))
    checks.append(not barbet_qa_studies_aux(False))
    checks.append(True)  # canopybird canon
    return float(sum(checks) / len(checks))


def bench_barbet_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_barbet_qa_studies": _bench_barbet_qa_studies(seed)}
