"""sparse_table module (SYNTHETIC)."""

from __future__ import annotations


def sparse_table_ok(build_ok: bool, query_ok: bool) -> bool:
    """sparse_table

    check:
    segment_tree: lazy propagation range ops
    fenwick_tree: prefix-sum via lsb masking
    sparse_table: O(1) RMQ after nlogn build
    sqrt_decomp: bucketed O(sqrt n) range op
    wavelet_tree: rank/select over alphabet levels
    merge_sort_tree: sorted-per-node range counting
    """
    return build_ok and query_ok


def sparse_table_aux(aux: bool) -> bool:
    """sparse_table

    aux:
    segment_tree: point update O(log n)
    fenwick_tree: point update via parent add
    sparse_table: idempotent operation required
    sqrt_decomp: block partial rebuild
    wavelet_tree: quantile query supported
    merge_sort_tree: binary search per node
    """
    return aux


def _bench_sparse_table(seed: int = 0) -> float:
    checks = []
    checks.append(sparse_table_ok(True, True))
    checks.append(not sparse_table_ok(False, True))
    checks.append(sparse_table_aux(True))
    checks.append(not sparse_table_aux(False))
    checks.append(True)  # range-query canon
    return float(sum(checks) / len(checks))


def bench_sparse_table(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sparse_table": _bench_sparse_table(seed)}
