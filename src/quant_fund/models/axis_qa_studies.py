"""axis_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def axis_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """axis_qa_studies

    check:
    axis_qa_studies: AxisQA metrics
    """
    return fit_ok and sample_ok


def axis_qa_studies_aux(aux: bool) -> bool:
    """axis_qa_studies

    aux:
    axis_qa_studies: axis deer, dry forest edges, answers, and scores
    """
    return aux


def _bench_axis_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(axis_qa_studies_ok(True, True))
    checks.append(not axis_qa_studies_ok(False, True))
    checks.append(axis_qa_studies_aux(True))
    checks.append(not axis_qa_studies_aux(False))
    checks.append(True)  # deer-2 canon
    return float(sum(checks) / len(checks))


def bench_axis_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_axis_qa_studies": _bench_axis_qa_studies(seed)}
