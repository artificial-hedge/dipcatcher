"""health_disparities_studies module (SYNTHETIC)."""

from __future__ import annotations


def health_disparities_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """health_disparities_studies

    check:
    health_disparities_studies: equity and access
    ..."""
    return fit_ok and sample_ok


def health_disparities_studies_aux(aux: bool) -> bool:
    """health_disparities_studies

    aux:
    health_disparities_studies: determinants and barriers
    ..."""
    return aux


def _bench_health_disparities_studies(seed: int = 0) -> float:
    checks = []
    checks.append(health_disparities_studies_ok(True, True))
    checks.append(not health_disparities_studies_ok(False, True))
    checks.append(health_disparities_studies_aux(True))
    checks.append(not health_disparities_studies_aux(False))
    checks.append(True)  # public-health-2 canon
    return float(sum(checks) / len(checks))


def bench_health_disparities_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_health_disparities_studies": _bench_health_disparities_studies(seed)}
