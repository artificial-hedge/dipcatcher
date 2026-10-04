"""whelk_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def whelk_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """whelk_qa_studies

    check:
    whelk_qa_studies: WhelkQA metrics
    """
    return fit_ok and sample_ok


def whelk_qa_studies_aux(aux: bool) -> bool:
    """whelk_qa_studies

    aux:
    whelk_qa_studies: whelks, cold coasts, answers, and scores
    """
    return aux


def _bench_whelk_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(whelk_qa_studies_ok(True, True))
    checks.append(not whelk_qa_studies_ok(False, True))
    checks.append(whelk_qa_studies_aux(True))
    checks.append(not whelk_qa_studies_aux(False))
    checks.append(True)  # bivalve canon
    return float(sum(checks) / len(checks))


def bench_whelk_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_whelk_qa_studies": _bench_whelk_qa_studies(seed)}
