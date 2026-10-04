"""puffball_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def puffball_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """puffball_qa_studies

    check:
    puffball_qa_studies: PuffballQA metrics
    """
    return fit_ok and sample_ok


def puffball_qa_studies_aux(aux: bool) -> bool:
    """puffball_qa_studies

    aux:
    puffball_qa_studies: puffballs, fields, answers, and scores
    """
    return aux


def _bench_puffball_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(puffball_qa_studies_ok(True, True))
    checks.append(not puffball_qa_studies_ok(False, True))
    checks.append(puffball_qa_studies_aux(True))
    checks.append(not puffball_qa_studies_aux(False))
    checks.append(True)  # fungi canon
    return float(sum(checks) / len(checks))


def bench_puffball_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_puffball_qa_studies": _bench_puffball_qa_studies(seed)}
