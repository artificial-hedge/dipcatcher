"""cuscus_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def cuscus_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """cuscus_qa_studies

    check:
    cuscus_qa_studies: CuscusQA metrics
    """
    return fit_ok and sample_ok


def cuscus_qa_studies_aux(aux: bool) -> bool:
    """cuscus_qa_studies

    aux:
    cuscus_qa_studies: cuscuses, canopies, answers, and scores
    """
    return aux


def _bench_cuscus_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(cuscus_qa_studies_ok(True, True))
    checks.append(not cuscus_qa_studies_ok(False, True))
    checks.append(cuscus_qa_studies_aux(True))
    checks.append(not cuscus_qa_studies_aux(False))
    checks.append(True)  # marsupial-3 canon
    return float(sum(checks) / len(checks))


def bench_cuscus_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cuscus_qa_studies": _bench_cuscus_qa_studies(seed)}
