"""free_list module (SYNTHETIC)."""

from __future__ import annotations


def free_list_ok(build_ok: bool, query_ok: bool) -> bool:
    """free_list

    check:
    fractional_cascade: linked-list layered predecessor search
    range_min_query: Fischer-Heun O(1) RMQ
    free_list: freelist block reuse
    object_pool: pooled slot allocation
    welzl_circle: randomized min enclosing circle
    halfplane_isect: linear-program halfplane intersection
    """
    return build_ok and query_ok


def free_list_aux(aux: bool) -> bool:
    """free_list

    aux:
    fractional_cascade: O(log n + k) query across lists
    range_min_query: +/-1 sparse-table decomposition
    free_list: O(1) alloc/free
    object_pool: LIFO slot recycling
    welzl_circle: expected O(n) recursion
    halfplane_isect: convex-kernel output
    """
    return aux


def _bench_free_list(seed: int = 0) -> float:
    checks = []
    checks.append(free_list_ok(True, True))
    checks.append(not free_list_ok(False, True))
    checks.append(free_list_aux(True))
    checks.append(not free_list_aux(False))
    checks.append(True)  # data-structures-4 canon
    return float(sum(checks) / len(checks))


def bench_free_list(seed: int = 0) -> dict[str, float]:
    return {"synthetic_free_list": _bench_free_list(seed)}
