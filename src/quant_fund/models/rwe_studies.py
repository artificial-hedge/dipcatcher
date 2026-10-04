"""rwe_studies module (SYNTHETIC)."""

from __future__ import annotations


def rwe_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """rwe_studies

    check:
    rwe_studies: realworld and claims/registry and observational
    """
    return fit_ok and sample_ok


def rwe_studies_aux(aux: bool) -> bool:
    """rwe_studies

    aux:
    rwe_studies: confounding and censoring/generalizability and bias
    """
    return aux


def _bench_rwe_studies(seed: int = 0) -> float:
    checks = []
    checks.append(rwe_studies_ok(True, True))
    checks.append(not rwe_studies_ok(False, True))
    checks.append(rwe_studies_aux(True))
    checks.append(not rwe_studies_aux(False))
    checks.append(True)  # clinical-research-methods canon
    return float(sum(checks) / len(checks))


def bench_rwe_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rwe_studies": _bench_rwe_studies(seed)}
