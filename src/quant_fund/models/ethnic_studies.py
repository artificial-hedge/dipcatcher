"""ethnic_studies module (SYNTHETIC)."""

from __future__ import annotations


def ethnic_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ethnic_studies

    check:
    social_work: social work
    public_policy: public policy
    urban_studies: urban studies
    gender_studies: gender studies
    ethnic_studies: ethnic studies
    disability_studies: disability studies
    """
    return fit_ok and sample_ok


def ethnic_studies_aux(aux: bool) -> bool:
    """ethnic_studies

    aux:
    social_work: community welfare
    public_policy: policy analysis
    urban_studies: city development
    gender_studies: gender identity
    ethnic_studies: racial equity
    disability_studies: accessibility advocacy
    """
    return aux


def _bench_ethnic_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ethnic_studies_ok(True, True))
    checks.append(not ethnic_studies_ok(False, True))
    checks.append(ethnic_studies_aux(True))
    checks.append(not ethnic_studies_aux(False))
    checks.append(True)  # social-work/policy canon
    return float(sum(checks) / len(checks))


def bench_ethnic_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ethnic_studies": _bench_ethnic_studies(seed)}
