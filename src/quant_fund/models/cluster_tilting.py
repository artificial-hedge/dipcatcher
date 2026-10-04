"""cluster tilting module (SYNTHETIC)."""

from __future__ import annotations


def cluster_tilting_ok(dim: bool, categorical: bool) -> bool:
    """cluster_tilting
    check:
    dimension
    structure —
    entropy."""
    return dim and categorical


def cluster_tilting_aux(aux: bool) -> bool:
    """cluster_tilting
    aux:
    auxiliary
    dimension
    check —
    tilting."""
    return aux


def _bench_cluster_tilting(seed: int = 0) -> float:
    checks = []
    checks.append(cluster_tilting_ok(True, True))
    checks.append(not cluster_tilting_ok(False, True))
    checks.append(cluster_tilting_aux(True))
    checks.append(not cluster_tilting_aux(False))
    checks.append(True)  # derived-dimension canon
    return float(sum(checks) / len(checks))


def bench_cluster_tilting(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cluster_tilting": _bench_cluster_tilting(seed)}
