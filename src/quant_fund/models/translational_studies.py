"""translational_studies module (SYNTHETIC)."""

from __future__ import annotations


def translational_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """translational_studies

    check:
    translational_studies: biomarkers and validation/analytical and clinical
    """
    return fit_ok and sample_ok


def translational_studies_aux(aux: bool) -> bool:
    """translational_studies

    aux:
    translational_studies: qualification and evidence/bench and bedside
    """
    return aux


def _bench_translational_studies(seed: int = 0) -> float:
    checks = []
    checks.append(translational_studies_ok(True, True))
    checks.append(not translational_studies_ok(False, True))
    checks.append(translational_studies_aux(True))
    checks.append(not translational_studies_aux(False))
    checks.append(True)  # trial-statistics/HEOR canon
    return float(sum(checks) / len(checks))


def bench_translational_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_translational_studies": _bench_translational_studies(seed)}
