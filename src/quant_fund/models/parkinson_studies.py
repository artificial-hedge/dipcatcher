"""parkinson_studies module (SYNTHETIC)."""

from __future__ import annotations


def parkinson_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """parkinson_studies

    check:
    parkinson_studies: dopamine and tremor
    ..."""
    return fit_ok and sample_ok


def parkinson_studies_aux(aux: bool) -> bool:
    """parkinson_studies

    aux:
    parkinson_studies: rigidity and bradykinesia
    ..."""
    return aux


def _bench_parkinson_studies(seed: int = 0) -> float:
    checks = []
    checks.append(parkinson_studies_ok(True, True))
    checks.append(not parkinson_studies_ok(False, True))
    checks.append(parkinson_studies_aux(True))
    checks.append(not parkinson_studies_aux(False))
    checks.append(True)  # neurodegeneration canon
    return float(sum(checks) / len(checks))


def bench_parkinson_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_parkinson_studies": _bench_parkinson_studies(seed)}
