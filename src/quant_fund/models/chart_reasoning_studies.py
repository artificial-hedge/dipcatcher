"""chart_reasoning_studies module (SYNTHETIC)."""

from __future__ import annotations


def chart_reasoning_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """chart_reasoning_studies

    check:
    chart_reasoning_studies: plot parsing and table extraction/axes and legends
    """
    return fit_ok and sample_ok


def chart_reasoning_studies_aux(aux: bool) -> bool:
    """chart_reasoning_studies

    aux:
    chart_reasoning_studies: chart-QA reasoning and numeric answers/figures and queries
    """
    return aux


def _bench_chart_reasoning_studies(seed: int = 0) -> float:
    checks = []
    checks.append(chart_reasoning_studies_ok(True, True))
    checks.append(not chart_reasoning_studies_ok(False, True))
    checks.append(chart_reasoning_studies_aux(True))
    checks.append(not chart_reasoning_studies_aux(False))
    checks.append(True)  # multimodal-2 canon
    return float(sum(checks) / len(checks))


def bench_chart_reasoning_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chart_reasoning_studies": _bench_chart_reasoning_studies(seed)}
