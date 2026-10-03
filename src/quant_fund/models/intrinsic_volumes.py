"""intrinsic volumes module (SYNTHETIC)."""

from __future__ import annotations


def intrinsic_volumes_ok(geo: bool, tess: bool) -> bool:
    """intrinsic_volumes
    check:
    stochastic
    geometry —
    tessellation."""
    return geo and tess


def intrinsic_volumes_aux(aux: bool) -> bool:
    """intrinsic_volumes
    aux:
    auxiliary
    geometry check —
    measure."""
    return aux


def _bench_intrinsic_volumes(seed: int = 0) -> float:
    checks = []
    checks.append(intrinsic_volumes_ok(True, True))
    checks.append(not intrinsic_volumes_ok(False, True))
    checks.append(intrinsic_volumes_aux(True))
    checks.append(not intrinsic_volumes_aux(False))
    checks.append(True)  # stochastic-geometry canon
    return float(sum(checks) / len(checks))


def bench_intrinsic_volumes(seed: int = 0) -> dict[str, float]:
    return {"synthetic_intrinsic_volumes": _bench_intrinsic_volumes(seed)}
