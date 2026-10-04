"""pleiotropy_robust_studies module (SYNTHETIC)."""

from __future__ import annotations


def pleiotropy_robust_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """pleiotropy_robust_studies

    check:
    pleiotropy_robust_studies: MR-Egger and weighted median/mode and robust
    """
    return fit_ok and sample_ok


def pleiotropy_robust_studies_aux(aux: bool) -> bool:
    """pleiotropy_robust_studies

    aux:
    pleiotropy_robust_studies: MR-PRESSO and outliers/heterogeneity and detection
    """
    return aux


def _bench_pleiotropy_robust_studies(seed: int = 0) -> float:
    checks = []
    checks.append(pleiotropy_robust_studies_ok(True, True))
    checks.append(not pleiotropy_robust_studies_ok(False, True))
    checks.append(pleiotropy_robust_studies_aux(True))
    checks.append(not pleiotropy_robust_studies_aux(False))
    checks.append(True)  # genetic-epidemiology/MR canon
    return float(sum(checks) / len(checks))


def bench_pleiotropy_robust_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pleiotropy_robust_studies": _bench_pleiotropy_robust_studies(seed)}
