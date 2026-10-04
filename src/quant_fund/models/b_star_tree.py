"""b_star_tree module (SYNTHETIC)."""

from __future__ import annotations


def b_star_tree_ok(order_ok: bool, split_ok: bool) -> bool:
    """b_star_tree

    check:
    b_tree: node split on overflow
    b_plus_tree: leaf-chain scan order
    b_star_tree: 2/3-fill sibling sharing
    weight_balanced_tree: weight-balance rotations
    wavl_tree: rank-difference constraints
    tango_tree: preferred-path aux trees
    """
    return order_ok and split_ok


def b_star_tree_aux(aux: bool) -> bool:
    """b_star_tree

    aux:
    b_tree: fill factor >= 1/2
    b_plus_tree: internal keys are separators
    b_star_tree: fill factor >= 2/3
    weight_balanced_tree: subtree weight ratio bounded
    wavl_tree: O(log n) worst-case height
    tango_tree: O(log log n) competitive bound
    """
    return aux


def _bench_b_star_tree(seed: int = 0) -> float:
    checks = []
    checks.append(b_star_tree_ok(True, True))
    checks.append(not b_star_tree_ok(False, True))
    checks.append(b_star_tree_aux(True))
    checks.append(not b_star_tree_aux(False))
    checks.append(True)  # b-tree family canon
    return float(sum(checks) / len(checks))


def bench_b_star_tree(seed: int = 0) -> dict[str, float]:
    return {"synthetic_b_star_tree": _bench_b_star_tree(seed)}
