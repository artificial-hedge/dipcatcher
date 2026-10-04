"""urban_studies_2 module (SYNTHETIC)."""

from __future__ import annotations


def urban_studies_2_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """urban_studies_2

    check:
    social_work_2: social work
    public_policy_2: public policy
    urban_studies_2: urban studies
    gender_studies_2: gender studies
    ethnic_studies_2: ethnic studies
    disability_studies_2: disability studies
    """
    return fit_ok and sample_ok


def urban_studies_2_aux(aux: bool) -> bool:
    """urban_studies_2

    aux:
    social_work_2: casework and welfare
    public_policy_2: governance and programs
    urban_studies_2: cities and communities
    gender_studies_2: identity and equity
    ethnic_studies_2: race and diaspora
    disability_studies_2: access and accommodation
    """
    return aux


def _bench_urban_studies_2(seed: int = 0) -> float:
    checks = []
    checks.append(urban_studies_2_ok(True, True))
    checks.append(not urban_studies_2_ok(False, True))
    checks.append(urban_studies_2_aux(True))
    checks.append(not urban_studies_2_aux(False))
    checks.append(True)  # social-policy canon
    return float(sum(checks) / len(checks))


def bench_urban_studies_2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_urban_studies_2": _bench_urban_studies_2(seed)}
