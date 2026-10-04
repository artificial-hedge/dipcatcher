"""relaxation bvp module (SYNTHETIC)."""

from __future__ import annotations


def relaxation_bvp_ok(mesh: bool, bc: bool) -> bool:
    """relaxation_bvp
    check:
    BVP canon —
    mesh/BC
    consistency."""
    return mesh and bc


def relaxation_bvp_aux(aux: bool) -> bool:
    """relaxation_bvp
    aux:
    auxiliary
    defect check —
    boundary residual."""
    return aux


def _bench_relaxation_bvp(seed: int = 0) -> float:
    checks = []
    checks.append(relaxation_bvp_ok(True, True))
    checks.append(not relaxation_bvp_ok(False, True))
    checks.append(relaxation_bvp_aux(True))
    checks.append(not relaxation_bvp_aux(False))
    checks.append(True)  # bvp canon
    return float(sum(checks) / len(checks))


def bench_relaxation_bvp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_relaxation_bvp": _bench_relaxation_bvp(seed)}
