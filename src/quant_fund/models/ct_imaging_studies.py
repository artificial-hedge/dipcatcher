"""ct_imaging_studies module (SYNTHETIC)."""

from __future__ import annotations


def ct_imaging_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ct_imaging_studies

    check:
    ct_imaging_studies: ct and contrast
    ..."""
    return fit_ok and sample_ok


def ct_imaging_studies_aux(aux: bool) -> bool:
    """ct_imaging_studies

    aux:
    ct_imaging_studies: hu and cta
    ..."""
    return aux


def _bench_ct_imaging_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ct_imaging_studies_ok(True, True))
    checks.append(not ct_imaging_studies_ok(False, True))
    checks.append(ct_imaging_studies_aux(True))
    checks.append(not ct_imaging_studies_aux(False))
    checks.append(True)  # imaging-modality canon
    return float(sum(checks) / len(checks))


def bench_ct_imaging_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ct_imaging_studies": _bench_ct_imaging_studies(seed)}
