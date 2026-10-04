"""chromatic base module (SYNTHETIC)."""

from __future__ import annotations


def chromatic_base_ok(chromatic: bool, stable: bool) -> bool:
    """chromatic_base
    check:
    chromatic
    structure —
    height."""
    return chromatic and stable


def chromatic_base_aux(aux: bool) -> bool:
    """chromatic_base
    aux:
    auxiliary
    chromatic
    check —
    tower."""
    return aux


def _bench_chromatic_base(seed: int = 0) -> float:
    checks = []
    checks.append(chromatic_base_ok(True, True))
    checks.append(not chromatic_base_ok(False, True))
    checks.append(chromatic_base_aux(True))
    checks.append(not chromatic_base_aux(False))
    checks.append(True)  # chromatic canon
    return float(sum(checks) / len(checks))


def bench_chromatic_base(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chromatic_base": _bench_chromatic_base(seed)}
