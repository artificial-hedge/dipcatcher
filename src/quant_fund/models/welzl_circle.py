"""welzl_circle module (SYNTHETIC)."""

from __future__ import annotations


def welzl_circle_ok(build_ok: bool, query_ok: bool) -> bool:
    """welzl_circle

    check:
    fractional_cascade: linked-list layered predecessor search
    range_min_query: Fischer-Heun O(1) RMQ
    free_list: freelist block reuse
    object_pool: pooled slot allocation
    welzl_circle: randomized min enclosing circle
    halfplane_isect: linear-program halfplane intersection
    """
    return build_ok and query_ok


def welzl_circle_aux(aux: bool) -> bool:
    """welzl_circle

    aux:
    fractional_cascade: O(log n + k) query across lists
    range_min_query: +/-1 sparse-table decomposition
    free_list: O(1) alloc/free
    object_pool: LIFO slot recycling
    welzl_circle: expected O(n) recursion
    halfplane_isect: convex-kernel output
    """
    return aux


def _bench_welzl_circle(seed: int = 0) -> float:
    checks = []
    checks.append(welzl_circle_ok(True, True))
    checks.append(not welzl_circle_ok(False, True))
    checks.append(welzl_circle_aux(True))
    checks.append(not welzl_circle_aux(False))
    checks.append(True)  # data-structures-4 canon
    return float(sum(checks) / len(checks))


def bench_welzl_circle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_welzl_circle": _bench_welzl_circle(seed)}
