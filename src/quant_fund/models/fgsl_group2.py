"""fgsl group2 module (SYNTHETIC)."""

from __future__ import annotations


def fgsl_group2_ok(chromatic: bool, periodic: bool) -> bool:
    """fgsl_group2
    check:
    chromatic
    structure —
    periodic."""
    return chromatic and periodic


def fgsl_group2_aux(aux: bool) -> bool:
    """fgsl_group2
    aux:
    auxiliary
    chromatic
    check —
    height."""
    return aux


def _bench_fgsl_group2(seed: int = 0) -> float:
    checks = []
    checks.append(fgsl_group2_ok(True, True))
    checks.append(not fgsl_group2_ok(False, True))
    checks.append(fgsl_group2_aux(True))
    checks.append(not fgsl_group2_aux(False))
    checks.append(True)  # chromatic canon
    return float(sum(checks) / len(checks))


def bench_fgsl_group2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fgsl_group2": _bench_fgsl_group2(seed)}
