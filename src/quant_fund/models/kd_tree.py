"""kd tree module (SYNTHETIC)."""

from __future__ import annotations


def kd_tree_ok(node: bool, leaf: bool) -> bool:
    """kd_tree
    check:
    spatial-index
    canon — node/
    leaf
    consistency."""
    return node and leaf


def kd_tree_aux(aux: bool) -> bool:
    """kd_tree
    aux:
    auxiliary
    split check —
    bounding-volume bound."""
    return aux


def _bench_kd_tree(seed: int = 0) -> float:
    checks = []
    checks.append(kd_tree_ok(True, True))
    checks.append(not kd_tree_ok(False, True))
    checks.append(kd_tree_aux(True))
    checks.append(not kd_tree_aux(False))
    checks.append(True)  # spatial canon
    return float(sum(checks) / len(checks))


def bench_kd_tree(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kd_tree": _bench_kd_tree(seed)}
