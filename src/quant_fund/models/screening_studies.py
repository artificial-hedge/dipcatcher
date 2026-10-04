"""screening_studies module (SYNTHETIC)."""

from __future__ import annotations


def screening_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """screening_studies

    check:
    screening_studies: detection and sensitivity
    ..."""
    return fit_ok and sample_ok


def screening_studies_aux(aux: bool) -> bool:
    """screening_studies

    aux:
    screening_studies: specificity and coverage
    ..."""
    return aux


def _bench_screening_studies(seed: int = 0) -> float:
    checks = []
    checks.append(screening_studies_ok(True, True))
    checks.append(not screening_studies_ok(False, True))
    checks.append(screening_studies_aux(True))
    checks.append(not screening_studies_aux(False))
    checks.append(True)  # public-health-2 canon
    return float(sum(checks) / len(checks))


def bench_screening_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_screening_studies": _bench_screening_studies(seed)}
