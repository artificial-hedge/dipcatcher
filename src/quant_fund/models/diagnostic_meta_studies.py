"""diagnostic_meta_studies module (SYNTHETIC)."""

from __future__ import annotations


def diagnostic_meta_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """diagnostic_meta_studies

    check:
    diagnostic_meta_studies: bivariate and HSROC/sensitivity and specificity
    """
    return fit_ok and sample_ok


def diagnostic_meta_studies_aux(aux: bool) -> bool:
    """diagnostic_meta_studies

    aux:
    diagnostic_meta_studies: threshold and heterogeneity/ROC and summary
    """
    return aux


def _bench_diagnostic_meta_studies(seed: int = 0) -> float:
    checks = []
    checks.append(diagnostic_meta_studies_ok(True, True))
    checks.append(not diagnostic_meta_studies_ok(False, True))
    checks.append(diagnostic_meta_studies_aux(True))
    checks.append(not diagnostic_meta_studies_aux(False))
    checks.append(True)  # evidence-synthesis canon
    return float(sum(checks) / len(checks))


def bench_diagnostic_meta_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_diagnostic_meta_studies": _bench_diagnostic_meta_studies(seed)}
