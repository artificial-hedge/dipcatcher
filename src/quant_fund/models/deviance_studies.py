"""deviance_studies module (SYNTHETIC)."""

from __future__ import annotations


def deviance_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """deviance_studies

    check:
    medical_sociology: medical sociology
    deviance_studies: deviance studies
    family_sociology: family sociology
    organization_theory: organization theory
    social_movements: social movements
    rural_sociology: rural sociology
    """
    return fit_ok and sample_ok


def deviance_studies_aux(aux: bool) -> bool:
    """deviance_studies

    aux:
    medical_sociology: health sociology
    deviance_studies: labeling theory
    family_sociology: kinship research
    organization_theory: institutional analysis
    social_movements: collective action
    rural_sociology: agrarian communities
    """
    return aux


def _bench_deviance_studies(seed: int = 0) -> float:
    checks = []
    checks.append(deviance_studies_ok(True, True))
    checks.append(not deviance_studies_ok(False, True))
    checks.append(deviance_studies_aux(True))
    checks.append(not deviance_studies_aux(False))
    checks.append(True)  # sociology-2 canon
    return float(sum(checks) / len(checks))


def bench_deviance_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_deviance_studies": _bench_deviance_studies(seed)}
