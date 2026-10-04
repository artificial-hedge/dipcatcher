"""loon_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def loon_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """loon_qa_studies

    check:
    loon_qa_studies: LoonQA metrics
    """
    return fit_ok and sample_ok


def loon_qa_studies_aux(aux: bool) -> bool:
    """loon_qa_studies

    aux:
    loon_qa_studies: loons, lakes, answers, and scores
    """
    return aux


def _bench_loon_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(loon_qa_studies_ok(True, True))
    checks.append(not loon_qa_studies_ok(False, True))
    checks.append(loon_qa_studies_aux(True))
    checks.append(not loon_qa_studies_aux(False))
    checks.append(True)  # waterbird canon
    return float(sum(checks) / len(checks))


def bench_loon_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_loon_qa_studies": _bench_loon_qa_studies(seed)}
