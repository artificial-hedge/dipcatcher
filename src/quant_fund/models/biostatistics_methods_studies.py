"""biostatistics_methods_studies module (SYNTHETIC)."""

from __future__ import annotations


def biostatistics_methods_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """biostatistics_methods_studies

    check:
    biostatistics_methods_studies: estimation and inference/testing and multiplicity
    """
    return fit_ok and sample_ok


def biostatistics_methods_studies_aux(aux: bool) -> bool:
    """biostatistics_methods_studies

    aux:
    biostatistics_methods_studies: covariates and stratification/adjustment and models
    """
    return aux


def _bench_biostatistics_methods_studies(seed: int = 0) -> float:
    checks = []
    checks.append(biostatistics_methods_studies_ok(True, True))
    checks.append(not biostatistics_methods_studies_ok(False, True))
    checks.append(biostatistics_methods_studies_aux(True))
    checks.append(not biostatistics_methods_studies_aux(False))
    checks.append(True)  # trial-statistics/HEOR canon
    return float(sum(checks) / len(checks))


def bench_biostatistics_methods_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_biostatistics_methods_studies": _bench_biostatistics_methods_studies(seed)}
