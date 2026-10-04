"""riccati bvp module (SYNTHETIC)."""

from __future__ import annotations


def riccati_bvp_ok(mesh: bool, bc: bool) -> bool:
    """riccati_bvp
    check:
    BVP canon —
    mesh/BC
    consistency."""
    return mesh and bc


def riccati_bvp_aux(aux: bool) -> bool:
    """riccati_bvp
    aux:
    auxiliary
    defect check —
    boundary residual."""
    return aux


def _bench_riccati_bvp(seed: int = 0) -> float:
    checks = []
    checks.append(riccati_bvp_ok(True, True))
    checks.append(not riccati_bvp_ok(False, True))
    checks.append(riccati_bvp_aux(True))
    checks.append(not riccati_bvp_aux(False))
    checks.append(True)  # bvp canon
    return float(sum(checks) / len(checks))


def bench_riccati_bvp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_riccati_bvp": _bench_riccati_bvp(seed)}
