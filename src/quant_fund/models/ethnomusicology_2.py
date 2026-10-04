"""ethnomusicology_2 module (SYNTHETIC)."""

from __future__ import annotations


def ethnomusicology_2_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ethnomusicology_2

    check:
    music_theory_2: music theory
    musicology_2: musicology
    ethnomusicology_2: ethnomusicology
    music_cognition_2: music cognition
    organology_2: organology
    composition_studies: composition studies
    """
    return fit_ok and sample_ok


def ethnomusicology_2_aux(aux: bool) -> bool:
    """ethnomusicology_2

    aux:
    music_theory_2: harmony and counterpoint
    musicology_2: repertoire and history
    ethnomusicology_2: culture and tradition
    music_cognition_2: perception and memory
    organology_2: instruments and construction
    composition_studies: scores and works
    """
    return aux


def _bench_ethnomusicology_2(seed: int = 0) -> float:
    checks = []
    checks.append(ethnomusicology_2_ok(True, True))
    checks.append(not ethnomusicology_2_ok(False, True))
    checks.append(ethnomusicology_2_aux(True))
    checks.append(not ethnomusicology_2_aux(False))
    checks.append(True)  # music canon
    return float(sum(checks) / len(checks))


def bench_ethnomusicology_2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ethnomusicology_2": _bench_ethnomusicology_2(seed)}
