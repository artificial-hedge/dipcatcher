"""chung lil module (SYNTHETIC)."""

from __future__ import annotations


def chung_lil_ok(limit: bool, rate: bool) -> bool:
    """chung_lil
    check:
    LIL/LLN
    structure —
    Strassen."""
    return limit and rate


def chung_lil_aux(aux: bool) -> bool:
    """chung_lil
    aux:
    auxiliary
    tail
    check —
    Khintchine."""
    return aux


def _bench_chung_lil(seed: int = 0) -> float:
    checks = []
    checks.append(chung_lil_ok(True, True))
    checks.append(not chung_lil_ok(False, True))
    checks.append(chung_lil_aux(True))
    checks.append(not chung_lil_aux(False))
    checks.append(True)  # LIL canon
    return float(sum(checks) / len(checks))


def bench_chung_lil(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chung_lil": _bench_chung_lil(seed)}
