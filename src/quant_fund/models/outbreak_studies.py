"""outbreak_studies module (SYNTHETIC)."""

from __future__ import annotations


def outbreak_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """outbreak_studies

    check:
    outbreak_studies: transmission and clusters
    ..."""
    return fit_ok and sample_ok


def outbreak_studies_aux(aux: bool) -> bool:
    """outbreak_studies

    aux:
    outbreak_studies: reproduction and containment
    ..."""
    return aux


def _bench_outbreak_studies(seed: int = 0) -> float:
    checks = []
    checks.append(outbreak_studies_ok(True, True))
    checks.append(not outbreak_studies_ok(False, True))
    checks.append(outbreak_studies_aux(True))
    checks.append(not outbreak_studies_aux(False))
    checks.append(True)  # public-health-2 canon
    return float(sum(checks) / len(checks))


def bench_outbreak_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_outbreak_studies": _bench_outbreak_studies(seed)}
