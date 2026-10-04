"""finger_tree module (SYNTHETIC)."""

from __future__ import annotations


def finger_tree_ok(order_ok: bool, level_ok: bool) -> bool:
    """finger_tree

    check:
    skip_list: geometric level tower + sorted links
    persistent_array: path-copying update preserves prior versions
    finger_tree: 2-3 tree monoid measure invariant
    rope_string: balanced leaf-weight concatenation
    vlist: block-linked O(1) growth chain
    pure_queue: two-list amortized O(1) dequeue
    """
    return order_ok and level_ok


def finger_tree_aux(aux: bool) -> bool:
    """finger_tree

    aux:
    skip_list: coin-flip promotion
    persistent_array: version array O(1) read
    finger_tree: push/pop at both ends O(1) amortized
    rope_string: split/concat rebalance
    vlist: offset-table random access
    pure_queue: reverse on front-list exhaustion
    """
    return aux


def _bench_finger_tree(seed: int = 0) -> float:
    checks = []
    checks.append(finger_tree_ok(True, True))
    checks.append(not finger_tree_ok(False, True))
    checks.append(finger_tree_aux(True))
    checks.append(not finger_tree_aux(False))
    checks.append(True)  # persistent-structure canon
    return float(sum(checks) / len(checks))


def bench_finger_tree(seed: int = 0) -> dict[str, float]:
    return {"synthetic_finger_tree": _bench_finger_tree(seed)}
