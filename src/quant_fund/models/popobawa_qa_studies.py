"""popobawa_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def popobawa_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """popobawa_qa_studies

    check:
    popobawa_qa_studies: PopobawaQA metrics
    """
    return fit_ok and sample_ok


def popobawa_qa_studies_aux(aux: bool) -> bool:
    """popobawa_qa_studies

    aux:
    popobawa_qa_studies: popobawas, bat shadows, answers, and scores
    """
    return aux


def _bench_popobawa_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(popobawa_qa_studies_ok(True, True))
    checks.append(not popobawa_qa_studies_ok(False, True))
    checks.append(popobawa_qa_studies_aux(True))
    checks.append(not popobawa_qa_studies_aux(False))
    checks.append(True)  # african-beast canon
    return float(sum(checks) / len(checks))


def bench_popobawa_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_popobawa_qa_studies": _bench_popobawa_qa_studies(seed)}
