"""ms_studies module (SYNTHETIC)."""

from __future__ import annotations


def ms_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ms_studies

    check:
    ms_studies: demyelination and plaques
    ..."""
    return fit_ok and sample_ok


def ms_studies_aux(aux: bool) -> bool:
    """ms_studies

    aux:
    ms_studies: lesions and oligoclonal
    ..."""
    return aux


def _bench_ms_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ms_studies_ok(True, True))
    checks.append(not ms_studies_ok(False, True))
    checks.append(ms_studies_aux(True))
    checks.append(not ms_studies_aux(False))
    checks.append(True)  # neurodegeneration canon
    return float(sum(checks) / len(checks))


def bench_ms_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ms_studies": _bench_ms_studies(seed)}
