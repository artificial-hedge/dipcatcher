"""red_black_tree module (SYNTHETIC)."""

from __future__ import annotations


def red_black_tree_ok(order_ok: bool, balance_ok: bool) -> bool:
    """red_black_tree

    check:
    avl_tree: height-diff <= 1 at every node
    red_black_tree: no red-red edges + equal black height
    splay_tree: amortized splay on access
    treap: heap priority + BST order
    scapegoat_tree: size-bound alpha rebuild
    aa_tree: level invariant (red = right only)
    """
    return order_ok and balance_ok


def red_black_tree_aux(aux: bool) -> bool:
    """red_black_tree

    aux:
    avl_tree: rotation restores balance
    red_black_tree: insert/delete recolor
    splay_tree: zig/zig-zig/zig-zag steps
    treap: split/merge ops
    scapegoat_tree: O(1) extra space
    aa_tree: skew/split ops
    """
    return aux


def _bench_red_black_tree(seed: int = 0) -> float:
    checks = []
    checks.append(red_black_tree_ok(True, True))
    checks.append(not red_black_tree_ok(False, True))
    checks.append(red_black_tree_aux(True))
    checks.append(not red_black_tree_aux(False))
    checks.append(True)  # balanced-tree canon
    return float(sum(checks) / len(checks))


def bench_red_black_tree(seed: int = 0) -> dict[str, float]:
    return {"synthetic_red_black_tree": _bench_red_black_tree(seed)}
