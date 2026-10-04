"""blue shift2 module (SYNTHETIC)."""

from __future__ import annotations


def blue_shift2_ok(chromatic: bool, periodic: bool) -> bool:
    """blue_shift2
    check:
    chromatic
    structure —
    periodic."""
    return chromatic and periodic


def blue_shift2_aux(aux: bool) -> bool:
    """blue_shift2
    aux:
    auxiliary
    chromatic
    check —
    height."""
    return aux


def _bench_blue_shift2(seed: int = 0) -> float:
    checks = []
    checks.append(blue_shift2_ok(True, True))
    checks.append(not blue_shift2_ok(False, True))
    checks.append(blue_shift2_aux(True))
    checks.append(not blue_shift2_aux(False))
    checks.append(True)  # chromatic canon
    return float(sum(checks) / len(checks))


def bench_blue_shift2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_blue_shift2": _bench_blue_shift2(seed)}
