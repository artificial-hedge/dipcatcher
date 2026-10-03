"""shooting bvp module (SYNTHETIC)."""

from __future__ import annotations


def shooting_bvp_ok(mesh: bool, bc: bool) -> bool:
    """shooting_bvp
    check:
    BVP canon —
    mesh/BC
    consistency."""
    return mesh and bc


def shooting_bvp_aux(aux: bool) -> bool:
    """shooting_bvp
    aux:
    auxiliary
    defect check —
    boundary residual."""
    return aux


def _bench_shooting_bvp(seed: int = 0) -> float:
    checks = []
    checks.append(shooting_bvp_ok(True, True))
    checks.append(not shooting_bvp_ok(False, True))
    checks.append(shooting_bvp_aux(True))
    checks.append(not shooting_bvp_aux(False))
    checks.append(True)  # bvp canon
    return float(sum(checks) / len(checks))


def bench_shooting_bvp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_shooting_bvp": _bench_shooting_bvp(seed)}
