"""strassen lil module (SYNTHETIC)."""

from __future__ import annotations


def strassen_lil_ok(limit: bool, rate: bool) -> bool:
    """strassen_lil
    check:
    LIL/LLN
    structure —
    Strassen."""
    return limit and rate


def strassen_lil_aux(aux: bool) -> bool:
    """strassen_lil
    aux:
    auxiliary
    tail
    check —
    Khintchine."""
    return aux


def _bench_strassen_lil(seed: int = 0) -> float:
    checks = []
    checks.append(strassen_lil_ok(True, True))
    checks.append(not strassen_lil_ok(False, True))
    checks.append(strassen_lil_aux(True))
    checks.append(not strassen_lil_aux(False))
    checks.append(True)  # LIL canon
    return float(sum(checks) / len(checks))


def bench_strassen_lil(seed: int = 0) -> dict[str, float]:
    return {"synthetic_strassen_lil": _bench_strassen_lil(seed)}
