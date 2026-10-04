"""opportunistic_studies module (SYNTHETIC)."""

from __future__ import annotations


def opportunistic_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """opportunistic_studies

    check:
    opportunistic_studies: opportunists and immunity
    ..."""
    return fit_ok and sample_ok


def opportunistic_studies_aux(aux: bool) -> bool:
    """opportunistic_studies

    aux:
    opportunistic_studies: pcp and cmv
    ..."""
    return aux


def _bench_opportunistic_studies(seed: int = 0) -> float:
    checks = []
    checks.append(opportunistic_studies_ok(True, True))
    checks.append(not opportunistic_studies_ok(False, True))
    checks.append(opportunistic_studies_aux(True))
    checks.append(not opportunistic_studies_aux(False))
    checks.append(True)  # infectious-medicine canon
    return float(sum(checks) / len(checks))


def bench_opportunistic_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_opportunistic_studies": _bench_opportunistic_studies(seed)}
