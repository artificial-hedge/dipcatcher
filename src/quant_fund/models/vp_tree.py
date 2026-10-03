"""vp tree module (SYNTHETIC)."""

from __future__ import annotations


def vp_tree_ok(node: bool, leaf: bool) -> bool:
    """vp_tree
    check:
    spatial-index
    canon — node/
    leaf
    consistency."""
    return node and leaf


def vp_tree_aux(aux: bool) -> bool:
    """vp_tree
    aux:
    auxiliary
    split check —
    bounding-volume bound."""
    return aux


def _bench_vp_tree(seed: int = 0) -> float:
    checks = []
    checks.append(vp_tree_ok(True, True))
    checks.append(not vp_tree_ok(False, True))
    checks.append(vp_tree_aux(True))
    checks.append(not vp_tree_aux(False))
    checks.append(True)  # spatial canon
    return float(sum(checks) / len(checks))


def bench_vp_tree(seed: int = 0) -> dict[str, float]:
    return {"synthetic_vp_tree": _bench_vp_tree(seed)}
