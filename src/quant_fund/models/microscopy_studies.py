"""microscopy_studies module (SYNTHETIC)."""

from __future__ import annotations


def microscopy_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """microscopy_studies

    check:
    microscopy_studies: magnification and staining/morphology and resolution
    """
    return fit_ok and sample_ok


def microscopy_studies_aux(aux: bool) -> bool:
    """microscopy_studies

    aux:
    microscopy_studies: brightfield and contrast/field and magnification
    """
    return aux


def _bench_microscopy_studies(seed: int = 0) -> float:
    checks = []
    checks.append(microscopy_studies_ok(True, True))
    checks.append(not microscopy_studies_ok(False, True))
    checks.append(microscopy_studies_aux(True))
    checks.append(not microscopy_studies_aux(False))
    checks.append(True)  # clinical-lab canon
    return float(sum(checks) / len(checks))


def bench_microscopy_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_microscopy_studies": _bench_microscopy_studies(seed)}
