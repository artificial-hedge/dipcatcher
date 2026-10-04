"""fir_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def fir_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fir_qa_studies

    check:
    fir_qa_studies: FirQA metrics
    """
    return fit_ok and sample_ok


def fir_qa_studies_aux(aux: bool) -> bool:
    """fir_qa_studies

    aux:
    fir_qa_studies: firs, needles, answers, and scores
    """
    return aux


def _bench_fir_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(fir_qa_studies_ok(True, True))
    checks.append(not fir_qa_studies_ok(False, True))
    checks.append(fir_qa_studies_aux(True))
    checks.append(not fir_qa_studies_aux(False))
    checks.append(True)  # evergreen canon
    return float(sum(checks) / len(checks))


def bench_fir_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fir_qa_studies": _bench_fir_qa_studies(seed)}
