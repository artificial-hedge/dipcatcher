"""psychopathology module (SYNTHETIC)."""

from __future__ import annotations


def psychopathology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """psychopathology

    check:
    experimental_psychology: experimental psychology
    comparative_psychology: comparative psychology
    evolutionary_psychology: evolutionary psychology
    psychopathology: psychopathology
    environmental_psychology: environmental psychology
    sport_psychology: sport psychology
    """
    return fit_ok and sample_ok


def psychopathology_aux(aux: bool) -> bool:
    """psychopathology

    aux:
    experimental_psychology: controlled studies
    comparative_psychology: cross-species behavior
    evolutionary_psychology: adaptive behavior
    psychopathology: mental disorders
    environmental_psychology: environment behavior
    sport_psychology: athletic performance
    """
    return aux


def _bench_psychopathology(seed: int = 0) -> float:
    checks = []
    checks.append(psychopathology_ok(True, True))
    checks.append(not psychopathology_ok(False, True))
    checks.append(psychopathology_aux(True))
    checks.append(not psychopathology_aux(False))
    checks.append(True)  # psychology-3 canon
    return float(sum(checks) / len(checks))


def bench_psychopathology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_psychopathology": _bench_psychopathology(seed)}
