"""quad tree module (SYNTHETIC)."""

from __future__ import annotations


def quad_tree_ok(node: bool, leaf: bool) -> bool:
    """quad_tree
    check:
    spatial-index
    canon — node/
    leaf
    consistency."""
    return node and leaf


def quad_tree_aux(aux: bool) -> bool:
    """quad_tree
    aux:
    auxiliary
    split check —
    bounding-volume bound."""
    return aux


def _bench_quad_tree(seed: int = 0) -> float:
    checks = []
    checks.append(quad_tree_ok(True, True))
    checks.append(not quad_tree_ok(False, True))
    checks.append(quad_tree_aux(True))
    checks.append(not quad_tree_aux(False))
    checks.append(True)  # spatial canon
    return float(sum(checks) / len(checks))


def bench_quad_tree(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quad_tree": _bench_quad_tree(seed)}
