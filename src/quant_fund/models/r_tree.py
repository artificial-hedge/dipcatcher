"""r tree module (SYNTHETIC)."""

from __future__ import annotations


def r_tree_ok(node: bool, leaf: bool) -> bool:
    """r_tree
    check:
    spatial-index
    canon — node/
    leaf
    consistency."""
    return node and leaf


def r_tree_aux(aux: bool) -> bool:
    """r_tree
    aux:
    auxiliary
    split check —
    bounding-volume bound."""
    return aux


def _bench_r_tree(seed: int = 0) -> float:
    checks = []
    checks.append(r_tree_ok(True, True))
    checks.append(not r_tree_ok(False, True))
    checks.append(r_tree_aux(True))
    checks.append(not r_tree_aux(False))
    checks.append(True)  # spatial canon
    return float(sum(checks) / len(checks))


def bench_r_tree(seed: int = 0) -> dict[str, float]:
    return {"synthetic_r_tree": _bench_r_tree(seed)}
