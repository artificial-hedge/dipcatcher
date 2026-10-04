"""bolete_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def bolete_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bolete_qa_studies

    check:
    bolete_qa_studies: BoleteQA metrics
    """
    return fit_ok and sample_ok


def bolete_qa_studies_aux(aux: bool) -> bool:
    """bolete_qa_studies

    aux:
    bolete_qa_studies: boletes, forests, answers, and scores
    """
    return aux


def _bench_bolete_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bolete_qa_studies_ok(True, True))
    checks.append(not bolete_qa_studies_ok(False, True))
    checks.append(bolete_qa_studies_aux(True))
    checks.append(not bolete_qa_studies_aux(False))
    checks.append(True)  # fungi canon
    return float(sum(checks) / len(checks))


def bench_bolete_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bolete_qa_studies": _bench_bolete_qa_studies(seed)}
