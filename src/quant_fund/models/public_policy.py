"""public_policy module (SYNTHETIC)."""

from __future__ import annotations


def public_policy_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """public_policy

    check:
    social_work: social work
    public_policy: public policy
    urban_studies: urban studies
    gender_studies: gender studies
    ethnic_studies: ethnic studies
    disability_studies: disability studies
    """
    return fit_ok and sample_ok


def public_policy_aux(aux: bool) -> bool:
    """public_policy

    aux:
    social_work: community welfare
    public_policy: policy analysis
    urban_studies: city development
    gender_studies: gender identity
    ethnic_studies: racial equity
    disability_studies: accessibility advocacy
    """
    return aux


def _bench_public_policy(seed: int = 0) -> float:
    checks = []
    checks.append(public_policy_ok(True, True))
    checks.append(not public_policy_ok(False, True))
    checks.append(public_policy_aux(True))
    checks.append(not public_policy_aux(False))
    checks.append(True)  # social-work/policy canon
    return float(sum(checks) / len(checks))


def bench_public_policy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_public_policy": _bench_public_policy(seed)}
