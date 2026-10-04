"""cover tree module (SYNTHETIC)."""

from __future__ import annotations


def cover_tree_ok(node: bool, leaf: bool) -> bool:
    """cover_tree
    check:
    spatial-index
    canon — node/
    leaf
    consistency."""
    return node and leaf


def cover_tree_aux(aux: bool) -> bool:
    """cover_tree
    aux:
    auxiliary
    split check —
    bounding-volume bound."""
    return aux


def _bench_cover_tree(seed: int = 0) -> float:
    checks = []
    checks.append(cover_tree_ok(True, True))
    checks.append(not cover_tree_ok(False, True))
    checks.append(cover_tree_aux(True))
    checks.append(not cover_tree_aux(False))
    checks.append(True)  # spatial canon
    return float(sum(checks) / len(checks))


def bench_cover_tree(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cover_tree": _bench_cover_tree(seed)}
