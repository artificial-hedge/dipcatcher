"""nist_metric_studies module (SYNTHETIC)."""

from __future__ import annotations


def nist_metric_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nist_metric_studies

    check:
    nist_metric_studies: NIST informative n-gram metrics
    """
    return fit_ok and sample_ok


def nist_metric_studies_aux(aux: bool) -> bool:
    """nist_metric_studies

    aux:
    nist_metric_studies: hypotheses, references, labels, and scores
    """
    return aux


def _bench_nist_metric_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nist_metric_studies_ok(True, True))
    checks.append(not nist_metric_studies_ok(False, True))
    checks.append(nist_metric_studies_aux(True))
    checks.append(not nist_metric_studies_aux(False))
    checks.append(True)  # translation-metric canon
    return float(sum(checks) / len(checks))


def bench_nist_metric_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nist_metric_studies": _bench_nist_metric_studies(seed)}
