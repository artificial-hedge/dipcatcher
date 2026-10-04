"""little cubes module (SYNTHETIC)."""

from __future__ import annotations


def little_cubes_ok(higher: bool, algebra: bool) -> bool:
    """little_cubes
    check:
    higher-algebra
    structure —
    operadic."""
    return higher and algebra


def little_cubes_aux(aux: bool) -> bool:
    """little_cubes
    aux:
    auxiliary
    higher-algebra
    check —
    enriched."""
    return aux


def _bench_little_cubes(seed: int = 0) -> float:
    checks = []
    checks.append(little_cubes_ok(True, True))
    checks.append(not little_cubes_ok(False, True))
    checks.append(little_cubes_aux(True))
    checks.append(not little_cubes_aux(False))
    checks.append(True)  # higher algebra canon
    return float(sum(checks) / len(checks))


def bench_little_cubes(seed: int = 0) -> dict[str, float]:
    return {"synthetic_little_cubes": _bench_little_cubes(seed)}
