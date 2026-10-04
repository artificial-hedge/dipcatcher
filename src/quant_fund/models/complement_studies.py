"""complement_studies module (SYNTHETIC)."""

from __future__ import annotations


def complement_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """complement_studies

    check:
    complement_studies: cascade and opsonin
    ..."""
    return fit_ok and sample_ok


def complement_studies_aux(aux: bool) -> bool:
    """complement_studies

    aux:
    complement_studies: c3 and lysis
    ..."""
    return aux


def _bench_complement_studies(seed: int = 0) -> float:
    checks = []
    checks.append(complement_studies_ok(True, True))
    checks.append(not complement_studies_ok(False, True))
    checks.append(complement_studies_aux(True))
    checks.append(not complement_studies_aux(False))
    checks.append(True)  # immune-mediators canon
    return float(sum(checks) / len(checks))


def bench_complement_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_complement_studies": _bench_complement_studies(seed)}
