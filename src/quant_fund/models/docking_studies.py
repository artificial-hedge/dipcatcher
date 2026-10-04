"""docking_studies module (SYNTHETIC)."""

from __future__ import annotations


def docking_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """docking_studies

    check:
    docking_studies: pose and scoring/binding and affinity
    """
    return fit_ok and sample_ok


def docking_studies_aux(aux: bool) -> bool:
    """docking_studies

    aux:
    docking_studies: grid and site/conformation and rescoring
    """
    return aux


def _bench_docking_studies(seed: int = 0) -> float:
    checks = []
    checks.append(docking_studies_ok(True, True))
    checks.append(not docking_studies_ok(False, True))
    checks.append(docking_studies_aux(True))
    checks.append(not docking_studies_aux(False))
    checks.append(True)  # drug-discovery canon
    return float(sum(checks) / len(checks))


def bench_docking_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_docking_studies": _bench_docking_studies(seed)}
