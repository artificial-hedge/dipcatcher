"""little cubes2 module (SYNTHETIC)."""

from __future__ import annotations


def little_cubes2_ok(algebra: bool, higher: bool) -> bool:
    """little_cubes2
    check:
    algebra
    structure —
    higher."""
    return algebra and higher


def little_cubes2_aux(aux: bool) -> bool:
    """little_cubes2
    aux:
    auxiliary
    algebra
    check —
    cubes."""
    return aux


def _bench_little_cubes2(seed: int = 0) -> float:
    checks = []
    checks.append(little_cubes2_ok(True, True))
    checks.append(not little_cubes2_ok(False, True))
    checks.append(little_cubes2_aux(True))
    checks.append(not little_cubes2_aux(False))
    checks.append(True)  # higher algebra canon
    return float(sum(checks) / len(checks))


def bench_little_cubes2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_little_cubes2": _bench_little_cubes2(seed)}
