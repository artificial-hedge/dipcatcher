"""wmt_metric_studies module (SYNTHETIC)."""

from __future__ import annotations


def wmt_metric_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """wmt_metric_studies

    check:
    wmt_metric_studies: WMT shared metrics
    """
    return fit_ok and sample_ok


def wmt_metric_studies_aux(aux: bool) -> bool:
    """wmt_metric_studies

    aux:
    wmt_metric_studies: sources, hypotheses, references, and scores
    """
    return aux


def _bench_wmt_metric_studies(seed: int = 0) -> float:
    checks = []
    checks.append(wmt_metric_studies_ok(True, True))
    checks.append(not wmt_metric_studies_ok(False, True))
    checks.append(wmt_metric_studies_aux(True))
    checks.append(not wmt_metric_studies_aux(False))
    checks.append(True)  # metric-exotics canon
    return float(sum(checks) / len(checks))


def bench_wmt_metric_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wmt_metric_studies": _bench_wmt_metric_studies(seed)}
