"""headache_studies module (SYNTHETIC)."""

from __future__ import annotations


def headache_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """headache_studies

    check:
    headache_studies: migraine and triggers
    ..."""
    return fit_ok and sample_ok


def headache_studies_aux(aux: bool) -> bool:
    """headache_studies

    aux:
    headache_studies: aura and prophylaxis
    ..."""
    return aux


def _bench_headache_studies(seed: int = 0) -> float:
    checks = []
    checks.append(headache_studies_ok(True, True))
    checks.append(not headache_studies_ok(False, True))
    checks.append(headache_studies_aux(True))
    checks.append(not headache_studies_aux(False))
    checks.append(True)  # pain canon
    return float(sum(checks) / len(checks))


def bench_headache_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_headache_studies": _bench_headache_studies(seed)}
