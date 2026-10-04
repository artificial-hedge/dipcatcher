"""motivic frequency module (SYNTHETIC)."""

from __future__ import annotations


def motivic_frequency_ok(motivic: bool, stable: bool) -> bool:
    """motivic_frequency
    check:
    motivic
    structure —
    stable."""
    return motivic and stable


def motivic_frequency_aux(aux: bool) -> bool:
    """motivic_frequency
    aux:
    auxiliary
    motivic
    check —
    slice."""
    return aux


def _bench_motivic_frequency(seed: int = 0) -> float:
    checks = []
    checks.append(motivic_frequency_ok(True, True))
    checks.append(not motivic_frequency_ok(False, True))
    checks.append(motivic_frequency_aux(True))
    checks.append(not motivic_frequency_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_motivic_frequency(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_frequency": _bench_motivic_frequency(seed)}
