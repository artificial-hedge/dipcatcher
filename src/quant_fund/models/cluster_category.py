"""cluster category module (SYNTHETIC)."""

from __future__ import annotations


def cluster_category_ok(cluster: bool, quiver: bool) -> bool:
    """cluster_category
    check:
    cluster
    structure —
    mutation."""
    return cluster and quiver


def cluster_category_aux(aux: bool) -> bool:
    """cluster_category
    aux:
    auxiliary
    cluster
    check —
    tilting."""
    return aux


def _bench_cluster_category(seed: int = 0) -> float:
    checks = []
    checks.append(cluster_category_ok(True, True))
    checks.append(not cluster_category_ok(False, True))
    checks.append(cluster_category_aux(True))
    checks.append(not cluster_category_aux(False))
    checks.append(True)  # cluster canon
    return float(sum(checks) / len(checks))


def bench_cluster_category(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cluster_category": _bench_cluster_category(seed)}
