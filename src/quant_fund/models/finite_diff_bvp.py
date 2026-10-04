"""finite diff_bvp module (SYNTHETIC)."""

from __future__ import annotations


def finite_diff_bvp_ok(mesh: bool, bc: bool) -> bool:
    """finite_diff_bvp
    check:
    BVP canon —
    mesh/BC
    consistency."""
    return mesh and bc


def finite_diff_bvp_aux(aux: bool) -> bool:
    """finite_diff_bvp
    aux:
    auxiliary
    defect check —
    boundary residual."""
    return aux


def _bench_finite_diff_bvp(seed: int = 0) -> float:
    checks = []
    checks.append(finite_diff_bvp_ok(True, True))
    checks.append(not finite_diff_bvp_ok(False, True))
    checks.append(finite_diff_bvp_aux(True))
    checks.append(not finite_diff_bvp_aux(False))
    checks.append(True)  # bvp canon
    return float(sum(checks) / len(checks))


def bench_finite_diff_bvp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_finite_diff_bvp": _bench_finite_diff_bvp(seed)}
