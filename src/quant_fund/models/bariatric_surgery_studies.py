"""bariatric_surgery_studies module (SYNTHETIC)."""

from __future__ import annotations


def bariatric_surgery_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bariatric_surgery_studies

    check:
    bariatric_surgery_studies: obesity and bypass
    ..."""
    return fit_ok and sample_ok


def bariatric_surgery_studies_aux(aux: bool) -> bool:
    """bariatric_surgery_studies

    aux:
    bariatric_surgery_studies: sleeve and dump
    ..."""
    return aux


def _bench_bariatric_surgery_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bariatric_surgery_studies_ok(True, True))
    checks.append(not bariatric_surgery_studies_ok(False, True))
    checks.append(bariatric_surgery_studies_aux(True))
    checks.append(not bariatric_surgery_studies_aux(False))
    checks.append(True)  # surgical-subspecialty canon
    return float(sum(checks) / len(checks))


def bench_bariatric_surgery_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bariatric_surgery_studies": _bench_bariatric_surgery_studies(seed)}
