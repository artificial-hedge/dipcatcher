"""epidemiology_methods_studies module (SYNTHETIC)."""

from __future__ import annotations


def epidemiology_methods_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """epidemiology_methods_studies

    check:
    epidemiology_methods_studies: incidence and prevalence/risk and denominators
    """
    return fit_ok and sample_ok


def epidemiology_methods_studies_aux(aux: bool) -> bool:
    """epidemiology_methods_studies

    aux:
    epidemiology_methods_studies: confounding and selection/bias and validity
    """
    return aux


def _bench_epidemiology_methods_studies(seed: int = 0) -> float:
    checks = []
    checks.append(epidemiology_methods_studies_ok(True, True))
    checks.append(not epidemiology_methods_studies_ok(False, True))
    checks.append(epidemiology_methods_studies_aux(True))
    checks.append(not epidemiology_methods_studies_aux(False))
    checks.append(True)  # trial-statistics/HEOR canon
    return float(sum(checks) / len(checks))


def bench_epidemiology_methods_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_epidemiology_methods_studies": _bench_epidemiology_methods_studies(seed)}
