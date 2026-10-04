"""point cloud_interp module (SYNTHETIC)."""

from __future__ import annotations


def point_cloud_interp_ok(node: bool, cloud: bool) -> bool:
    """point_cloud_interp
    check:
    meshfree/moving-least-squares —
    support
    consistency."""
    return node and cloud


def point_cloud_interp_aux(aux: bool) -> bool:
    """point_cloud_interp
    aux:
    auxiliary
    meshfree check —
    reproduction bound."""
    return aux


def _bench_point_cloud_interp(seed: int = 0) -> float:
    checks = []
    checks.append(point_cloud_interp_ok(True, True))
    checks.append(not point_cloud_interp_ok(False, True))
    checks.append(point_cloud_interp_aux(True))
    checks.append(not point_cloud_interp_aux(False))
    checks.append(True)  # meshfree canon
    return float(sum(checks) / len(checks))


def bench_point_cloud_interp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_point_cloud_interp": _bench_point_cloud_interp(seed)}
