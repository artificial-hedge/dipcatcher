"""refractive_studies module (SYNTHETIC)."""

from __future__ import annotations


def refractive_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """refractive_studies

    check:
    refractive_studies: myopia and astigmatism
    ..."""
    return fit_ok and sample_ok


def refractive_studies_aux(aux: bool) -> bool:
    """refractive_studies

    aux:
    refractive_studies: lasik and diopter
    ..."""
    return aux


def _bench_refractive_studies(seed: int = 0) -> float:
    checks = []
    checks.append(refractive_studies_ok(True, True))
    checks.append(not refractive_studies_ok(False, True))
    checks.append(refractive_studies_aux(True))
    checks.append(not refractive_studies_aux(False))
    checks.append(True)  # ophthalmology-vision canon
    return float(sum(checks) / len(checks))


def bench_refractive_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_refractive_studies": _bench_refractive_studies(seed)}
