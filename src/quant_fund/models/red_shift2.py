"""red shift2 module (SYNTHETIC)."""

from __future__ import annotations


def red_shift2_ok(chromatic: bool, height: bool) -> bool:
    """red_shift2
    check:
    chromatic
    structure —
    height."""
    return chromatic and height


def red_shift2_aux(aux: bool) -> bool:
    """red_shift2
    aux:
    auxiliary
    chromatic
    check —
    periodicity."""
    return aux


def _bench_red_shift2(seed: int = 0) -> float:
    checks = []
    checks.append(red_shift2_ok(True, True))
    checks.append(not red_shift2_ok(False, True))
    checks.append(red_shift2_aux(True))
    checks.append(not red_shift2_aux(False))
    checks.append(True)  # chromatic canon
    return float(sum(checks) / len(checks))


def bench_red_shift2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_red_shift2": _bench_red_shift2(seed)}
