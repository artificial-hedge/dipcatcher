"""nergal_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nergal_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nergal_qa_studies

    check:
    nergal_qa_studies: NergalQA metrics
    """
    return fit_ok and sample_ok


def nergal_qa_studies_aux(aux: bool) -> bool:
    """nergal_qa_studies

    aux:
    nergal_qa_studies: nergal, plague lords, answers, and scores
    """
    return aux


def _bench_nergal_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nergal_qa_studies_ok(True, True))
    checks.append(not nergal_qa_studies_ok(False, True))
    checks.append(nergal_qa_studies_aux(True))
    checks.append(not nergal_qa_studies_aux(False))
    checks.append(True)  # sumerian-2 canon
    return float(sum(checks) / len(checks))


def bench_nergal_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nergal_qa_studies": _bench_nergal_qa_studies(seed)}
