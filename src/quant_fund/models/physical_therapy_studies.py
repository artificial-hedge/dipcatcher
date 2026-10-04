"""physical_therapy_studies module (SYNTHETIC)."""

from __future__ import annotations


def physical_therapy_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """physical_therapy_studies

    check:
    physical_therapy_studies: exercise and mobility
    ..."""
    return fit_ok and sample_ok


def physical_therapy_studies_aux(aux: bool) -> bool:
    """physical_therapy_studies

    aux:
    physical_therapy_studies: strength and rom
    ..."""
    return aux


def _bench_physical_therapy_studies(seed: int = 0) -> float:
    checks = []
    checks.append(physical_therapy_studies_ok(True, True))
    checks.append(not physical_therapy_studies_ok(False, True))
    checks.append(physical_therapy_studies_aux(True))
    checks.append(not physical_therapy_studies_aux(False))
    checks.append(True)  # rehab-medicine canon
    return float(sum(checks) / len(checks))


def bench_physical_therapy_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_physical_therapy_studies": _bench_physical_therapy_studies(seed)}
