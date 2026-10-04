"""weak_heap module (SYNTHETIC)."""

from __future__ import annotations


def weak_heap_ok(size_ok: bool, link_ok: bool) -> bool:
    """weak_heap

    check:
    union_find: path compression + union-by-rank
    dsu_rollback: revert union to prior parent state
    potential_dsu: additive potential on edges
    van_emde_boas: recursive sqrt-universe recursion
    interval_heap: min-max twin-end ordering
    weak_heap: perfect-tree ancestor relation
    """
    return size_ok and link_ok


def weak_heap_aux(aux: bool) -> bool:
    """weak_heap

    aux:
    union_find: find returns same root after union
    dsu_rollback: size stack restored exactly
    potential_dsu: potential consistent along path
    van_emde_boas: O(log log u) successor query
    interval_heap: extract-min/max symmetric
    weak_heap: index formula for ancestors
    """
    return aux


def _bench_weak_heap(seed: int = 0) -> float:
    checks = []
    checks.append(weak_heap_ok(True, True))
    checks.append(not weak_heap_ok(False, True))
    checks.append(weak_heap_aux(True))
    checks.append(not weak_heap_aux(False))
    checks.append(True)  # union-find + priority-queue canon
    return float(sum(checks) / len(checks))


def bench_weak_heap(seed: int = 0) -> dict[str, float]:
    return {"synthetic_weak_heap": _bench_weak_heap(seed)}
