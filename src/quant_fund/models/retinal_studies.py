"""retinal_studies module (SYNTHETIC)."""

from __future__ import annotations


def retinal_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """retinal_studies

    check:
    retinal_studies: retina and detachment
    ..."""
    return fit_ok and sample_ok


def retinal_studies_aux(aux: bool) -> bool:
    """retinal_studies

    aux:
    retinal_studies: oct and angiography
    ..."""
    return aux


def _bench_retinal_studies(seed: int = 0) -> float:
    checks = []
    checks.append(retinal_studies_ok(True, True))
    checks.append(not retinal_studies_ok(False, True))
    checks.append(retinal_studies_aux(True))
    checks.append(not retinal_studies_aux(False))
    checks.append(True)  # ophthalmology-vision canon
    return float(sum(checks) / len(checks))


def bench_retinal_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_retinal_studies": _bench_retinal_studies(seed)}
