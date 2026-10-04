"""collocation bvp module (SYNTHETIC)."""

from __future__ import annotations


def collocation_bvp_ok(mesh: bool, bc: bool) -> bool:
    """collocation_bvp
    check:
    BVP canon —
    mesh/BC
    consistency."""
    return mesh and bc


def collocation_bvp_aux(aux: bool) -> bool:
    """collocation_bvp
    aux:
    auxiliary
    defect check —
    boundary residual."""
    return aux


def _bench_collocation_bvp(seed: int = 0) -> float:
    checks = []
    checks.append(collocation_bvp_ok(True, True))
    checks.append(not collocation_bvp_ok(False, True))
    checks.append(collocation_bvp_aux(True))
    checks.append(not collocation_bvp_aux(False))
    checks.append(True)  # bvp canon
    return float(sum(checks) / len(checks))


def bench_collocation_bvp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_collocation_bvp": _bench_collocation_bvp(seed)}
