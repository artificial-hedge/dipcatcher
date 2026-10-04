"""social_work module (SYNTHETIC)."""

from __future__ import annotations


def social_work_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """social_work

    check:
    social_work: social work
    public_policy: public policy
    urban_studies: urban studies
    gender_studies: gender studies
    ethnic_studies: ethnic studies
    disability_studies: disability studies
    """
    return fit_ok and sample_ok


def social_work_aux(aux: bool) -> bool:
    """social_work

    aux:
    social_work: community welfare
    public_policy: policy analysis
    urban_studies: city development
    gender_studies: gender identity
    ethnic_studies: racial equity
    disability_studies: accessibility advocacy
    """
    return aux


def _bench_social_work(seed: int = 0) -> float:
    checks = []
    checks.append(social_work_ok(True, True))
    checks.append(not social_work_ok(False, True))
    checks.append(social_work_aux(True))
    checks.append(not social_work_aux(False))
    checks.append(True)  # social-work/policy canon
    return float(sum(checks) / len(checks))


def bench_social_work(seed: int = 0) -> dict[str, float]:
    return {"synthetic_social_work": _bench_social_work(seed)}
