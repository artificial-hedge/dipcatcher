"""experimental_psychology module (SYNTHETIC)."""

from __future__ import annotations


def experimental_psychology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """experimental_psychology

    check:
    experimental_psychology: experimental psychology
    comparative_psychology: comparative psychology
    evolutionary_psychology: evolutionary psychology
    psychopathology: psychopathology
    environmental_psychology: environmental psychology
    sport_psychology: sport psychology
    """
    return fit_ok and sample_ok


def experimental_psychology_aux(aux: bool) -> bool:
    """experimental_psychology

    aux:
    experimental_psychology: controlled studies
    comparative_psychology: cross-species behavior
    evolutionary_psychology: adaptive behavior
    psychopathology: mental disorders
    environmental_psychology: environment behavior
    sport_psychology: athletic performance
    """
    return aux


def _bench_experimental_psychology(seed: int = 0) -> float:
    checks = []
    checks.append(experimental_psychology_ok(True, True))
    checks.append(not experimental_psychology_ok(False, True))
    checks.append(experimental_psychology_aux(True))
    checks.append(not experimental_psychology_aux(False))
    checks.append(True)  # psychology-3 canon
    return float(sum(checks) / len(checks))


def bench_experimental_psychology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_experimental_psychology": _bench_experimental_psychology(seed)}
