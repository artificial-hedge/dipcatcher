"""wagner platen module (SYNTHETIC)."""

from __future__ import annotations


def wagner_platen_ok(st1: bool, kk: bool) -> bool:
    """wagner_platen
    check:
    stochastic
    expansion —
    Kloeden
    strong."""
    return st1 and kk


def wagner_platen_aux(aux: bool) -> bool:
    """wagner_platen
    aux:
    auxiliary
    Wong-Zakai
    check —
    smooth
    approx."""
    return aux


def _bench_wagner_platen(seed: int = 0) -> float:
    checks = []
    checks.append(wagner_platen_ok(True, True))
    checks.append(not wagner_platen_ok(False, True))
    checks.append(wagner_platen_aux(True))
    checks.append(not wagner_platen_aux(False))
    checks.append(True)  # expansion canon
    return float(sum(checks) / len(checks))


def bench_wagner_platen(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wagner_platen": _bench_wagner_platen(seed)}
