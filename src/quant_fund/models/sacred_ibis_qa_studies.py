"""sacred_ibis_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def sacred_ibis_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sacred_ibis_qa_studies

    check:
    sacred_ibis_qa_studies: Sacred-ibisQA metrics
    """
    return fit_ok and sample_ok


def sacred_ibis_qa_studies_aux(aux: bool) -> bool:
    """sacred_ibis_qa_studies

    aux:
    sacred_ibis_qa_studies: sacred ibises, riverines, answers, and scores
    """
    return aux


def _bench_sacred_ibis_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sacred_ibis_qa_studies_ok(True, True))
    checks.append(not sacred_ibis_qa_studies_ok(False, True))
    checks.append(sacred_ibis_qa_studies_aux(True))
    checks.append(not sacred_ibis_qa_studies_aux(False))
    checks.append(True)  # egret canon
    return float(sum(checks) / len(checks))


def bench_sacred_ibis_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sacred_ibis_qa_studies": _bench_sacred_ibis_qa_studies(seed)}
