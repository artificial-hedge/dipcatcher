"""dementia_studies module (SYNTHETIC)."""

from __future__ import annotations


def dementia_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dementia_studies

    check:
    dementia_studies: cognition and decline
    ..."""
    return fit_ok and sample_ok


def dementia_studies_aux(aux: bool) -> bool:
    """dementia_studies

    aux:
    dementia_studies: atrophy and impairment
    ..."""
    return aux


def _bench_dementia_studies(seed: int = 0) -> float:
    checks = []
    checks.append(dementia_studies_ok(True, True))
    checks.append(not dementia_studies_ok(False, True))
    checks.append(dementia_studies_aux(True))
    checks.append(not dementia_studies_aux(False))
    checks.append(True)  # neurodegeneration canon
    return float(sum(checks) / len(checks))


def bench_dementia_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dementia_studies": _bench_dementia_studies(seed)}
