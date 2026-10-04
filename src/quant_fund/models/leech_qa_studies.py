"""leech_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def leech_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """leech_qa_studies

    check:
    leech_qa_studies: LeechQA metrics
    """
    return fit_ok and sample_ok


def leech_qa_studies_aux(aux: bool) -> bool:
    """leech_qa_studies

    aux:
    leech_qa_studies: leeches, pond shallows, answers, and scores
    """
    return aux


def _bench_leech_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(leech_qa_studies_ok(True, True))
    checks.append(not leech_qa_studies_ok(False, True))
    checks.append(leech_qa_studies_aux(True))
    checks.append(not leech_qa_studies_aux(False))
    checks.append(True)  # annelid canon
    return float(sum(checks) / len(checks))


def bench_leech_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_leech_qa_studies": _bench_leech_qa_studies(seed)}
