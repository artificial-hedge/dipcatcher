"""alloy_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def alloy_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """alloy_qa_studies

    check:
    alloy_qa_studies: AlloyQA metrics
    """
    return fit_ok and sample_ok


def alloy_qa_studies_aux(aux: bool) -> bool:
    """alloy_qa_studies

    aux:
    alloy_qa_studies: alloys, compositions, answers, and scores
    """
    return aux


def _bench_alloy_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(alloy_qa_studies_ok(True, True))
    checks.append(not alloy_qa_studies_ok(False, True))
    checks.append(alloy_qa_studies_aux(True))
    checks.append(not alloy_qa_studies_aux(False))
    checks.append(True)  # material canon
    return float(sum(checks) / len(checks))


def bench_alloy_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_alloy_qa_studies": _bench_alloy_qa_studies(seed)}
