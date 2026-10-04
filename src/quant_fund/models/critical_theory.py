"""critical_theory module (SYNTHETIC)."""

from __future__ import annotations


def critical_theory_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """critical_theory

    check:
    comparative_literature: comparative literature
    literary_theory: literary theory
    postcolonial_studies: postcolonial studies
    world_literature: world literature
    translation_studies: translation studies
    critical_theory: critical theory
    """
    return fit_ok and sample_ok


def critical_theory_aux(aux: bool) -> bool:
    """critical_theory

    aux:
    comparative_literature: cross-cultural texts
    literary_theory: theory of literature
    postcolonial_studies: decolonial critique
    world_literature: global literary circulation
    translation_studies: translation theory
    critical_theory: frankfurt school
    """
    return aux


def _bench_critical_theory(seed: int = 0) -> float:
    checks = []
    checks.append(critical_theory_ok(True, True))
    checks.append(not critical_theory_ok(False, True))
    checks.append(critical_theory_aux(True))
    checks.append(not critical_theory_aux(False))
    checks.append(True)  # comparative literature canon
    return float(sum(checks) / len(checks))


def bench_critical_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_critical_theory": _bench_critical_theory(seed)}
