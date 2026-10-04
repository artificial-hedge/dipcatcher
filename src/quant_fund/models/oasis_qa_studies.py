"""oasis_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def oasis_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """oasis_qa_studies

    check:
    oasis_qa_studies: OasisQA metrics
    """
    return fit_ok and sample_ok


def oasis_qa_studies_aux(aux: bool) -> bool:
    """oasis_qa_studies

    aux:
    oasis_qa_studies: oases, springs, answers, and scores
    """
    return aux


def _bench_oasis_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(oasis_qa_studies_ok(True, True))
    checks.append(not oasis_qa_studies_ok(False, True))
    checks.append(oasis_qa_studies_aux(True))
    checks.append(not oasis_qa_studies_aux(False))
    checks.append(True)  # desert-2 canon
    return float(sum(checks) / len(checks))


def bench_oasis_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_oasis_qa_studies": _bench_oasis_qa_studies(seed)}
