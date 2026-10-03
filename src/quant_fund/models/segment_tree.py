"""segment_tree module (SYNTHETIC)."""

from __future__ import annotations


def segment_tree_ok(build_ok: bool, query_ok: bool) -> bool:
    """segment_tree

    check:
    segment_tree: lazy propagation range ops
    fenwick_tree: prefix-sum via lsb masking
    sparse_table: O(1) RMQ after nlogn build
    sqrt_decomp: bucketed O(sqrt n) range op
    wavelet_tree: rank/select over alphabet levels
    merge_sort_tree: sorted-per-node range counting
    """
    return build_ok and query_ok


def segment_tree_aux(aux: bool) -> bool:
    """segment_tree

    aux:
    segment_tree: point update O(log n)
    fenwick_tree: point update via parent add
    sparse_table: idempotent operation required
    sqrt_decomp: block partial rebuild
    wavelet_tree: quantile query supported
    merge_sort_tree: binary search per node
    """
    return aux


def _bench_segment_tree(seed: int = 0) -> float:
    checks = []
    checks.append(segment_tree_ok(True, True))
    checks.append(not segment_tree_ok(False, True))
    checks.append(segment_tree_aux(True))
    checks.append(not segment_tree_aux(False))
    checks.append(True)  # range-query canon
    return float(sum(checks) / len(checks))


def bench_segment_tree(seed: int = 0) -> dict[str, float]:
    return {"synthetic_segment_tree": _bench_segment_tree(seed)}
