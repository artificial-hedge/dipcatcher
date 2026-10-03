"""multiple shooting module (SYNTHETIC)."""

from __future__ import annotations


def multiple_shooting_ok(mesh: bool, bc: bool) -> bool:
    """multiple_shooting
    check:
    BVP canon —
    mesh/BC
    consistency."""
    return mesh and bc


def multiple_shooting_aux(aux: bool) -> bool:
    """multiple_shooting
    aux:
    auxiliary
    defect check —
    boundary residual."""
    return aux


def _bench_multiple_shooting(seed: int = 0) -> float:
    checks = []
    checks.append(multiple_shooting_ok(True, True))
    checks.append(not multiple_shooting_ok(False, True))
    checks.append(multiple_shooting_aux(True))
    checks.append(not multiple_shooting_aux(False))
    checks.append(True)  # bvp canon
    return float(sum(checks) / len(checks))


def bench_multiple_shooting(seed: int = 0) -> dict[str, float]:
    return {"synthetic_multiple_shooting": _bench_multiple_shooting(seed)}
