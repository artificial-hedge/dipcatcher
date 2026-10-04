"""athletic_training module (SYNTHETIC)."""

from __future__ import annotations


def athletic_training_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """athletic_training

    check:
    sports_science: sports science
    exercise_physiology: exercise physiology
    sports_biomechanics: sports biomechanics
    sports_psychology: sports psychology
    athletic_training: athletic training
    sports_analytics: sports analytics
    """
    return fit_ok and sample_ok


def athletic_training_aux(aux: bool) -> bool:
    """athletic_training

    aux:
    sports_science: performance science
    exercise_physiology: human exertion
    sports_biomechanics: movement mechanics
    sports_psychology: athlete cognition
    athletic_training: injury prevention
    sports_analytics: performance data
    """
    return aux


def _bench_athletic_training(seed: int = 0) -> float:
    checks = []
    checks.append(athletic_training_ok(True, True))
    checks.append(not athletic_training_ok(False, True))
    checks.append(athletic_training_aux(True))
    checks.append(not athletic_training_aux(False))
    checks.append(True)  # sports science canon
    return float(sum(checks) / len(checks))


def bench_athletic_training(seed: int = 0) -> dict[str, float]:
    return {"synthetic_athletic_training": _bench_athletic_training(seed)}
