"""grouper_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def grouper_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """grouper_qa_studies

    check:
    grouper_qa_studies: GrouperQA metrics
    """
    return fit_ok and sample_ok


def grouper_qa_studies_aux(aux: bool) -> bool:
    """grouper_qa_studies

    aux:
    grouper_qa_studies: groupers, rocky outcrops, answers, and scores
    """
    return aux


def _bench_grouper_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(grouper_qa_studies_ok(True, True))
    checks.append(not grouper_qa_studies_ok(False, True))
    checks.append(grouper_qa_studies_aux(True))
    checks.append(not grouper_qa_studies_aux(False))
    checks.append(True)  # reef-fish canon
    return float(sum(checks) / len(checks))


def bench_grouper_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_grouper_qa_studies": _bench_grouper_qa_studies(seed)}
