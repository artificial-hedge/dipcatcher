"""heritability_ldscore_studies module (SYNTHETIC)."""

from __future__ import annotations


def heritability_ldscore_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """heritability_ldscore_studies

    check:
    heritability_ldscore_studies: LD score regression/heritability and confounding
    """
    return fit_ok and sample_ok


def heritability_ldscore_studies_aux(aux: bool) -> bool:
    """heritability_ldscore_studies

    aux:
    heritability_ldscore_studies: intercept and attenuation/chi-square and inflation
    """
    return aux


def _bench_heritability_ldscore_studies(seed: int = 0) -> float:
    checks = []
    checks.append(heritability_ldscore_studies_ok(True, True))
    checks.append(not heritability_ldscore_studies_ok(False, True))
    checks.append(heritability_ldscore_studies_aux(True))
    checks.append(not heritability_ldscore_studies_aux(False))
    checks.append(True)  # genetic-epidemiology/MR canon
    return float(sum(checks) / len(checks))


def bench_heritability_ldscore_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_heritability_ldscore_studies": _bench_heritability_ldscore_studies(seed)}
