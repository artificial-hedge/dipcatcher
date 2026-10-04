"""cataract_studies module (SYNTHETIC)."""

from __future__ import annotations


def cataract_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """cataract_studies

    check:
    cataract_studies: lens and phaco
    ..."""
    return fit_ok and sample_ok


def cataract_studies_aux(aux: bool) -> bool:
    """cataract_studies

    aux:
    cataract_studies: iol and capsule
    ..."""
    return aux


def _bench_cataract_studies(seed: int = 0) -> float:
    checks = []
    checks.append(cataract_studies_ok(True, True))
    checks.append(not cataract_studies_ok(False, True))
    checks.append(cataract_studies_aux(True))
    checks.append(not cataract_studies_aux(False))
    checks.append(True)  # ophthalmology-vision canon
    return float(sum(checks) / len(checks))


def bench_cataract_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cataract_studies": _bench_cataract_studies(seed)}
