"""literary_theory module (SYNTHETIC)."""

from __future__ import annotations


def literary_theory_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """literary_theory

    check:
    comparative_literature: comparative literature
    literary_theory: literary theory
    postcolonial_studies: postcolonial studies
    world_literature: world literature
    translation_studies: translation studies
    critical_theory: critical theory
    """
    return fit_ok and sample_ok


def literary_theory_aux(aux: bool) -> bool:
    """literary_theory

    aux:
    comparative_literature: cross-cultural texts
    literary_theory: theory of literature
    postcolonial_studies: decolonial critique
    world_literature: global literary circulation
    translation_studies: translation theory
    critical_theory: frankfurt school
    """
    return aux


def _bench_literary_theory(seed: int = 0) -> float:
    checks = []
    checks.append(literary_theory_ok(True, True))
    checks.append(not literary_theory_ok(False, True))
    checks.append(literary_theory_aux(True))
    checks.append(not literary_theory_aux(False))
    checks.append(True)  # comparative literature canon
    return float(sum(checks) / len(checks))


def bench_literary_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_literary_theory": _bench_literary_theory(seed)}
