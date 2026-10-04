"""data_quality_studies module (SYNTHETIC)."""

from __future__ import annotations


def data_quality_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """data_quality_studies

    check:
    data_quality_studies: heuristic filtering and perplexity scoring/rules and thresholds
    """
    return fit_ok and sample_ok


def data_quality_studies_aux(aux: bool) -> bool:
    """data_quality_studies

    aux:
    data_quality_studies: quality classifiers and document scoring/precision and recall
    """
    return aux


def _bench_data_quality_studies(seed: int = 0) -> float:
    checks = []
    checks.append(data_quality_studies_ok(True, True))
    checks.append(not data_quality_studies_ok(False, True))
    checks.append(data_quality_studies_aux(True))
    checks.append(not data_quality_studies_aux(False))
    checks.append(True)  # pretraining-data canon
    return float(sum(checks) / len(checks))


def bench_data_quality_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_data_quality_studies": _bench_data_quality_studies(seed)}
