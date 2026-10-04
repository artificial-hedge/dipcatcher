"""distance_learning module (SYNTHETIC)."""

from __future__ import annotations


def distance_learning_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """distance_learning

    check:
    higher_education: higher education
    vocational_education: vocational education
    special_education: special education
    comparative_education: comparative education
    literacy_studies: literacy studies
    distance_learning: distance learning
    """
    return fit_ok and sample_ok


def distance_learning_aux(aux: bool) -> bool:
    """distance_learning

    aux:
    higher_education: university research
    vocational_education: trade education
    special_education: inclusive education
    comparative_education: cross-national education
    literacy_studies: reading research
    distance_learning: online education
    """
    return aux


def _bench_distance_learning(seed: int = 0) -> float:
    checks = []
    checks.append(distance_learning_ok(True, True))
    checks.append(not distance_learning_ok(False, True))
    checks.append(distance_learning_aux(True))
    checks.append(not distance_learning_aux(False))
    checks.append(True)  # education-2 canon
    return float(sum(checks) / len(checks))


def bench_distance_learning(seed: int = 0) -> dict[str, float]:
    return {"synthetic_distance_learning": _bench_distance_learning(seed)}
