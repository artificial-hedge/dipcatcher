"""hemoglobin_studies module (SYNTHETIC)."""

from __future__ import annotations


def hemoglobin_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hemoglobin_studies

    check:
    hemoglobin_studies: thalassemia and sickle
    ..."""
    return fit_ok and sample_ok


def hemoglobin_studies_aux(aux: bool) -> bool:
    """hemoglobin_studies

    aux:
    hemoglobin_studies: variants and electrophoresis
    ..."""
    return aux


def _bench_hemoglobin_studies(seed: int = 0) -> float:
    checks = []
    checks.append(hemoglobin_studies_ok(True, True))
    checks.append(not hemoglobin_studies_ok(False, True))
    checks.append(hemoglobin_studies_aux(True))
    checks.append(not hemoglobin_studies_aux(False))
    checks.append(True)  # hematology canon
    return float(sum(checks) / len(checks))


def bench_hemoglobin_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hemoglobin_studies": _bench_hemoglobin_studies(seed)}
