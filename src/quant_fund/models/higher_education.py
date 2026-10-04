"""higher_education module (SYNTHETIC)."""

from __future__ import annotations


def higher_education_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """higher_education

    check:
    higher_education: higher education
    vocational_education: vocational education
    special_education: special education
    comparative_education: comparative education
    literacy_studies: literacy studies
    distance_learning: distance learning
    """
    return fit_ok and sample_ok


def higher_education_aux(aux: bool) -> bool:
    """higher_education

    aux:
    higher_education: university research
    vocational_education: trade education
    special_education: inclusive education
    comparative_education: cross-national education
    literacy_studies: reading research
    distance_learning: online education
    """
    return aux


def _bench_higher_education(seed: int = 0) -> float:
    checks = []
    checks.append(higher_education_ok(True, True))
    checks.append(not higher_education_ok(False, True))
    checks.append(higher_education_aux(True))
    checks.append(not higher_education_aux(False))
    checks.append(True)  # education-2 canon
    return float(sum(checks) / len(checks))


def bench_higher_education(seed: int = 0) -> dict[str, float]:
    return {"synthetic_higher_education": _bench_higher_education(seed)}
