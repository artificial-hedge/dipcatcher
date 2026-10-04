"""chart_qa_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def chart_qa_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """chart_qa_lite_studies

    check:
    chart_qa_lite_studies: ChartQA metrics
    """
    return fit_ok and sample_ok


def chart_qa_lite_studies_aux(aux: bool) -> bool:
    """chart_qa_lite_studies

    aux:
    chart_qa_lite_studies: charts, questions, answers, and scores
    """
    return aux


def _bench_chart_qa_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(chart_qa_lite_studies_ok(True, True))
    checks.append(not chart_qa_lite_studies_ok(False, True))
    checks.append(chart_qa_lite_studies_aux(True))
    checks.append(not chart_qa_lite_studies_aux(False))
    checks.append(True)  # vision-doc-QA canon
    return float(sum(checks) / len(checks))


def bench_chart_qa_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chart_qa_lite_studies": _bench_chart_qa_lite_studies(seed)}
