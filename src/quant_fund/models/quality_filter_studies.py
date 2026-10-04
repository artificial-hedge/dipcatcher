"""quality_filter_studies module (SYNTHETIC)."""

from __future__ import annotations


def quality_filter_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """quality_filter_studies

    check:
    quality_filter_studies: heuristic quality signals and gates/ratios and rules
    """
    return fit_ok and sample_ok


def quality_filter_studies_aux(aux: bool) -> bool:
    """quality_filter_studies

    aux:
    quality_filter_studies: Gopher/C4-style composite filters/metrics and verdicts
    """
    return aux


def _bench_quality_filter_studies(seed: int = 0) -> float:
    checks = []
    checks.append(quality_filter_studies_ok(True, True))
    checks.append(not quality_filter_studies_ok(False, True))
    checks.append(quality_filter_studies_aux(True))
    checks.append(not quality_filter_studies_aux(False))
    checks.append(True)  # data-filtering/dedup canon
    return float(sum(checks) / len(checks))


def bench_quality_filter_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quality_filter_studies": _bench_quality_filter_studies(seed)}
