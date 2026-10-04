"""social_movements module (SYNTHETIC)."""

from __future__ import annotations


def social_movements_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """social_movements

    check:
    medical_sociology: medical sociology
    deviance_studies: deviance studies
    family_sociology: family sociology
    organization_theory: organization theory
    social_movements: social movements
    rural_sociology: rural sociology
    """
    return fit_ok and sample_ok


def social_movements_aux(aux: bool) -> bool:
    """social_movements

    aux:
    medical_sociology: health sociology
    deviance_studies: labeling theory
    family_sociology: kinship research
    organization_theory: institutional analysis
    social_movements: collective action
    rural_sociology: agrarian communities
    """
    return aux


def _bench_social_movements(seed: int = 0) -> float:
    checks = []
    checks.append(social_movements_ok(True, True))
    checks.append(not social_movements_ok(False, True))
    checks.append(social_movements_aux(True))
    checks.append(not social_movements_aux(False))
    checks.append(True)  # sociology-2 canon
    return float(sum(checks) / len(checks))


def bench_social_movements(seed: int = 0) -> dict[str, float]:
    return {"synthetic_social_movements": _bench_social_movements(seed)}
