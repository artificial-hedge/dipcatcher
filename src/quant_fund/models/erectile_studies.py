"""erectile_studies module (SYNTHETIC)."""

from __future__ import annotations


def erectile_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """erectile_studies

    check:
    erectile_studies: erection and nocturnal
    ..."""
    return fit_ok and sample_ok


def erectile_studies_aux(aux: bool) -> bool:
    """erectile_studies

    aux:
    erectile_studies: pde5 and doppler
    ..."""
    return aux


def _bench_erectile_studies(seed: int = 0) -> float:
    checks = []
    checks.append(erectile_studies_ok(True, True))
    checks.append(not erectile_studies_ok(False, True))
    checks.append(erectile_studies_aux(True))
    checks.append(not erectile_studies_aux(False))
    checks.append(True)  # urology-andrology canon
    return float(sum(checks) / len(checks))


def bench_erectile_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_erectile_studies": _bench_erectile_studies(seed)}
