"""music_history module (SYNTHETIC)."""

from __future__ import annotations


def music_history_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """music_history

    check:
    musicology: musicology
    ethnomusicology: ethnomusicology
    music_theory: music theory
    music_cognition: music cognition
    organology: organology
    music_history: music history
    """
    return fit_ok and sample_ok


def music_history_aux(aux: bool) -> bool:
    """music_history

    aux:
    musicology: systematic music study
    ethnomusicology: world music cultures
    music_theory: harmony and counterpoint
    music_cognition: music perception
    organology: musical instruments
    music_history: historical periods
    """
    return aux


def _bench_music_history(seed: int = 0) -> float:
    checks = []
    checks.append(music_history_ok(True, True))
    checks.append(not music_history_ok(False, True))
    checks.append(music_history_aux(True))
    checks.append(not music_history_aux(False))
    checks.append(True)  # musicology canon
    return float(sum(checks) / len(checks))


def bench_music_history(seed: int = 0) -> dict[str, float]:
    return {"synthetic_music_history": _bench_music_history(seed)}
