"""comparative_psychology module (SYNTHETIC)."""

from __future__ import annotations


def comparative_psychology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """comparative_psychology

    check:
    experimental_psychology: experimental psychology
    comparative_psychology: comparative psychology
    evolutionary_psychology: evolutionary psychology
    psychopathology: psychopathology
    environmental_psychology: environmental psychology
    sport_psychology: sport psychology
    """
    return fit_ok and sample_ok


def comparative_psychology_aux(aux: bool) -> bool:
    """comparative_psychology

    aux:
    experimental_psychology: controlled studies
    comparative_psychology: cross-species behavior
    evolutionary_psychology: adaptive behavior
    psychopathology: mental disorders
    environmental_psychology: environment behavior
    sport_psychology: athletic performance
    """
    return aux


def _bench_comparative_psychology(seed: int = 0) -> float:
    checks = []
    checks.append(comparative_psychology_ok(True, True))
    checks.append(not comparative_psychology_ok(False, True))
    checks.append(comparative_psychology_aux(True))
    checks.append(not comparative_psychology_aux(False))
    checks.append(True)  # psychology-3 canon
    return float(sum(checks) / len(checks))


def bench_comparative_psychology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_comparative_psychology": _bench_comparative_psychology(seed)}
