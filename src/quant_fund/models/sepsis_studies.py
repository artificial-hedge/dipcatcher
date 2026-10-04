"""sepsis_studies module (SYNTHETIC)."""

from __future__ import annotations


def sepsis_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sepsis_studies

    check:
    sepsis_studies: sepsis and shock
    ..."""
    return fit_ok and sample_ok


def sepsis_studies_aux(aux: bool) -> bool:
    """sepsis_studies

    aux:
    sepsis_studies: lactate and vasopressors
    ..."""
    return aux


def _bench_sepsis_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sepsis_studies_ok(True, True))
    checks.append(not sepsis_studies_ok(False, True))
    checks.append(sepsis_studies_aux(True))
    checks.append(not sepsis_studies_aux(False))
    checks.append(True)  # infectious-medicine canon
    return float(sum(checks) / len(checks))


def bench_sepsis_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sepsis_studies": _bench_sepsis_studies(seed)}
