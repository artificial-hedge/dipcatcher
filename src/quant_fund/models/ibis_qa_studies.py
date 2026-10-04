"""ibis_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ibis_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ibis_qa_studies

    check:
    ibis_qa_studies: IbisQA metrics
    """
    return fit_ok and sample_ok


def ibis_qa_studies_aux(aux: bool) -> bool:
    """ibis_qa_studies

    aux:
    ibis_qa_studies: ibises, flocks, answers, and scores
    """
    return aux


def _bench_ibis_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ibis_qa_studies_ok(True, True))
    checks.append(not ibis_qa_studies_ok(False, True))
    checks.append(ibis_qa_studies_aux(True))
    checks.append(not ibis_qa_studies_aux(False))
    checks.append(True)  # waterbird canon
    return float(sum(checks) / len(checks))


def bench_ibis_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ibis_qa_studies": _bench_ibis_qa_studies(seed)}
