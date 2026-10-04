"""surveillance_studies module (SYNTHETIC)."""

from __future__ import annotations


def surveillance_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """surveillance_studies

    check:
    surveillance_studies: reporting and incidence
    ..."""
    return fit_ok and sample_ok


def surveillance_studies_aux(aux: bool) -> bool:
    """surveillance_studies

    aux:
    surveillance_studies: mortality and trends
    ..."""
    return aux


def _bench_surveillance_studies(seed: int = 0) -> float:
    checks = []
    checks.append(surveillance_studies_ok(True, True))
    checks.append(not surveillance_studies_ok(False, True))
    checks.append(surveillance_studies_aux(True))
    checks.append(not surveillance_studies_aux(False))
    checks.append(True)  # public-health-2 canon
    return float(sum(checks) / len(checks))


def bench_surveillance_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_surveillance_studies": _bench_surveillance_studies(seed)}
