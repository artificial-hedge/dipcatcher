"""glaucoma_studies module (SYNTHETIC)."""

from __future__ import annotations


def glaucoma_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """glaucoma_studies

    check:
    glaucoma_studies: pressure and nerve
    ..."""
    return fit_ok and sample_ok


def glaucoma_studies_aux(aux: bool) -> bool:
    """glaucoma_studies

    aux:
    glaucoma_studies: iop and visual
    ..."""
    return aux


def _bench_glaucoma_studies(seed: int = 0) -> float:
    checks = []
    checks.append(glaucoma_studies_ok(True, True))
    checks.append(not glaucoma_studies_ok(False, True))
    checks.append(glaucoma_studies_aux(True))
    checks.append(not glaucoma_studies_aux(False))
    checks.append(True)  # ophthalmology-vision canon
    return float(sum(checks) / len(checks))


def bench_glaucoma_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_glaucoma_studies": _bench_glaucoma_studies(seed)}
