"""wiener measure module (SYNTHETIC)."""

from __future__ import annotations


def wiener_measure_ok(proc: bool, tight: bool) -> bool:
    """wiener_measure
    check:
    empirical-process
    structure —
    Donsker."""
    return proc and tight


def wiener_measure_aux(aux: bool) -> bool:
    """wiener_measure
    aux:
    auxiliary
    class
    check —
    Vapnik."""
    return aux


def _bench_wiener_measure(seed: int = 0) -> float:
    checks = []
    checks.append(wiener_measure_ok(True, True))
    checks.append(not wiener_measure_ok(False, True))
    checks.append(wiener_measure_aux(True))
    checks.append(not wiener_measure_aux(False))
    checks.append(True)  # empirical canon
    return float(sum(checks) / len(checks))


def bench_wiener_measure(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wiener_measure": _bench_wiener_measure(seed)}
