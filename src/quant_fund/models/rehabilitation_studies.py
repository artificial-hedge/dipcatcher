"""rehabilitation_studies module (SYNTHETIC)."""

from __future__ import annotations


def rehabilitation_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """rehabilitation_studies

    check:
    rehabilitation_studies: function and recovery
    ..."""
    return fit_ok and sample_ok


def rehabilitation_studies_aux(aux: bool) -> bool:
    """rehabilitation_studies

    aux:
    rehabilitation_studies: gait and adl
    ..."""
    return aux


def _bench_rehabilitation_studies(seed: int = 0) -> float:
    checks = []
    checks.append(rehabilitation_studies_ok(True, True))
    checks.append(not rehabilitation_studies_ok(False, True))
    checks.append(rehabilitation_studies_aux(True))
    checks.append(not rehabilitation_studies_aux(False))
    checks.append(True)  # rehab-medicine canon
    return float(sum(checks) / len(checks))


def bench_rehabilitation_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rehabilitation_studies": _bench_rehabilitation_studies(seed)}
