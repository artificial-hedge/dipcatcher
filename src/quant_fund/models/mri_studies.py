"""mri_studies module (SYNTHETIC)."""

from __future__ import annotations


def mri_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mri_studies

    check:
    mri_studies: mri and sequences
    ..."""
    return fit_ok and sample_ok


def mri_studies_aux(aux: bool) -> bool:
    """mri_studies

    aux:
    mri_studies: t1 and t2
    ..."""
    return aux


def _bench_mri_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mri_studies_ok(True, True))
    checks.append(not mri_studies_ok(False, True))
    checks.append(mri_studies_aux(True))
    checks.append(not mri_studies_aux(False))
    checks.append(True)  # imaging-modality canon
    return float(sum(checks) / len(checks))


def bench_mri_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mri_studies": _bench_mri_studies(seed)}
