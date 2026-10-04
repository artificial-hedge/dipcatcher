"""chromatic l3 module (SYNTHETIC)."""

from __future__ import annotations


def chromatic_l3_ok(chromatic: bool, height: bool) -> bool:
    """chromatic_l3
    check:
    chromatic
    structure —
    height."""
    return chromatic and height


def chromatic_l3_aux(aux: bool) -> bool:
    """chromatic_l3
    aux:
    auxiliary
    chromatic
    check —
    periodicity."""
    return aux


def _bench_chromatic_l3(seed: int = 0) -> float:
    checks = []
    checks.append(chromatic_l3_ok(True, True))
    checks.append(not chromatic_l3_ok(False, True))
    checks.append(chromatic_l3_aux(True))
    checks.append(not chromatic_l3_aux(False))
    checks.append(True)  # chromatic canon
    return float(sum(checks) / len(checks))


def bench_chromatic_l3(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chromatic_l3": _bench_chromatic_l3(seed)}
