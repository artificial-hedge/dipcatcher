"""flora_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def flora_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """flora_qa_studies

    check:
    flora_qa_studies: FloraQA metrics
    """
    return fit_ok and sample_ok


def flora_qa_studies_aux(aux: bool) -> bool:
    """flora_qa_studies

    aux:
    flora_qa_studies: flora, spring blooms, answers, and scores
    """
    return aux


def _bench_flora_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(flora_qa_studies_ok(True, True))
    checks.append(not flora_qa_studies_ok(False, True))
    checks.append(flora_qa_studies_aux(True))
    checks.append(not flora_qa_studies_aux(False))
    checks.append(True)  # roman-rural canon
    return float(sum(checks) / len(checks))


def bench_flora_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_flora_qa_studies": _bench_flora_qa_studies(seed)}
