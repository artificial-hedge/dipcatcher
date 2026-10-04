"""age_bias_studies module (SYNTHETIC)."""

from __future__ import annotations


def age_bias_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """age_bias_studies

    check:
    age_bias_studies: Age-bias probe metrics
    """
    return fit_ok and sample_ok


def age_bias_studies_aux(aux: bool) -> bool:
    """age_bias_studies

    aux:
    age_bias_studies: texts, groups, labels, and accuracies
    """
    return aux


def _bench_age_bias_studies(seed: int = 0) -> float:
    checks = []
    checks.append(age_bias_studies_ok(True, True))
    checks.append(not age_bias_studies_ok(False, True))
    checks.append(age_bias_studies_aux(True))
    checks.append(not age_bias_studies_aux(False))
    checks.append(True)  # rumor-bias canon
    return float(sum(checks) / len(checks))


def bench_age_bias_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_age_bias_studies": _bench_age_bias_studies(seed)}
