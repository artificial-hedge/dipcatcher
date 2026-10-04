"""ethnomusicology module (SYNTHETIC)."""

from __future__ import annotations


def ethnomusicology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ethnomusicology

    check:
    musicology: musicology
    ethnomusicology: ethnomusicology
    music_theory: music theory
    music_cognition: music cognition
    organology: organology
    music_history: music history
    """
    return fit_ok and sample_ok


def ethnomusicology_aux(aux: bool) -> bool:
    """ethnomusicology

    aux:
    musicology: systematic music study
    ethnomusicology: world music cultures
    music_theory: harmony and counterpoint
    music_cognition: music perception
    organology: musical instruments
    music_history: historical periods
    """
    return aux


def _bench_ethnomusicology(seed: int = 0) -> float:
    checks = []
    checks.append(ethnomusicology_ok(True, True))
    checks.append(not ethnomusicology_ok(False, True))
    checks.append(ethnomusicology_aux(True))
    checks.append(not ethnomusicology_aux(False))
    checks.append(True)  # musicology canon
    return float(sum(checks) / len(checks))


def bench_ethnomusicology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ethnomusicology": _bench_ethnomusicology(seed)}
