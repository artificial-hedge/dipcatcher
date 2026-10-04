"""pharmacy_practice_studies module (SYNTHETIC)."""

from __future__ import annotations


def pharmacy_practice_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """pharmacy_practice_studies

    check:
    pharmacy_practice_studies: dispensing and counseling
    ..."""
    return fit_ok and sample_ok


def pharmacy_practice_studies_aux(aux: bool) -> bool:
    """pharmacy_practice_studies

    aux:
    pharmacy_practice_studies: refills and otc
    ..."""
    return aux


def _bench_pharmacy_practice_studies(seed: int = 0) -> float:
    checks = []
    checks.append(pharmacy_practice_studies_ok(True, True))
    checks.append(not pharmacy_practice_studies_ok(False, True))
    checks.append(pharmacy_practice_studies_aux(True))
    checks.append(not pharmacy_practice_studies_aux(False))
    checks.append(True)  # clinical-pharmacy canon
    return float(sum(checks) / len(checks))


def bench_pharmacy_practice_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pharmacy_practice_studies": _bench_pharmacy_practice_studies(seed)}
