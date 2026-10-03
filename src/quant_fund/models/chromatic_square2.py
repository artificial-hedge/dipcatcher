"""chromatic square2 module (SYNTHETIC)."""

from __future__ import annotations


def chromatic_square2_ok(chromatic: bool, stable: bool) -> bool:
    """chromatic_square2
    check:
    chromatic
    structure —
    height."""
    return chromatic and stable


def chromatic_square2_aux(aux: bool) -> bool:
    """chromatic_square2
    aux:
    auxiliary
    chromatic
    check —
    tower."""
    return aux


def _bench_chromatic_square2(seed: int = 0) -> float:
    checks = []
    checks.append(chromatic_square2_ok(True, True))
    checks.append(not chromatic_square2_ok(False, True))
    checks.append(chromatic_square2_aux(True))
    checks.append(not chromatic_square2_aux(False))
    checks.append(True)  # chromatic canon
    return float(sum(checks) / len(checks))


def bench_chromatic_square2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chromatic_square2": _bench_chromatic_square2(seed)}
