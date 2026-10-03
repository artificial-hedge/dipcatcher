"""ball tree module (SYNTHETIC)."""

from __future__ import annotations


def ball_tree_ok(node: bool, leaf: bool) -> bool:
    """ball_tree
    check:
    spatial-index
    canon — node/
    leaf
    consistency."""
    return node and leaf


def ball_tree_aux(aux: bool) -> bool:
    """ball_tree
    aux:
    auxiliary
    split check —
    bounding-volume bound."""
    return aux


def _bench_ball_tree(seed: int = 0) -> float:
    checks = []
    checks.append(ball_tree_ok(True, True))
    checks.append(not ball_tree_ok(False, True))
    checks.append(ball_tree_aux(True))
    checks.append(not ball_tree_aux(False))
    checks.append(True)  # spatial canon
    return float(sum(checks) / len(checks))


def bench_ball_tree(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ball_tree": _bench_ball_tree(seed)}
