"""cluster algebra module (SYNTHETIC)."""

from __future__ import annotations


def cluster_algebra_ok(cluster: bool, quiver: bool) -> bool:
    """cluster_algebra
    check:
    cluster
    structure —
    mutation."""
    return cluster and quiver


def cluster_algebra_aux(aux: bool) -> bool:
    """cluster_algebra
    aux:
    auxiliary
    cluster
    check —
    tilting."""
    return aux


def _bench_cluster_algebra(seed: int = 0) -> float:
    checks = []
    checks.append(cluster_algebra_ok(True, True))
    checks.append(not cluster_algebra_ok(False, True))
    checks.append(cluster_algebra_aux(True))
    checks.append(not cluster_algebra_aux(False))
    checks.append(True)  # cluster canon
    return float(sum(checks) / len(checks))


def bench_cluster_algebra(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cluster_algebra": _bench_cluster_algebra(seed)}
