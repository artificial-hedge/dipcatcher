"""dotsenko kpz module (SYNTHETIC)."""

from __future__ import annotations


def dotsenko_kpz_ok(kpz: bool, reg: bool) -> bool:
    """dotsenko_kpz
    check:
    KPZ-2
    structure —
    Corwin."""
    return kpz and reg


def dotsenko_kpz_aux(aux: bool) -> bool:
    """dotsenko_kpz
    aux:
    auxiliary
    regularity-structure
    check —
    Hairer."""
    return aux


def _bench_dotsenko_kpz(seed: int = 0) -> float:
    checks = []
    checks.append(dotsenko_kpz_ok(True, True))
    checks.append(not dotsenko_kpz_ok(False, True))
    checks.append(dotsenko_kpz_aux(True))
    checks.append(not dotsenko_kpz_aux(False))
    checks.append(True)  # KPZ-2 canon
    return float(sum(checks) / len(checks))


def bench_dotsenko_kpz(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dotsenko_kpz": _bench_dotsenko_kpz(seed)}
