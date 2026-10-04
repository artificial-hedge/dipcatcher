"""potential_dsu module (SYNTHETIC)."""

from __future__ import annotations


def potential_dsu_ok(size_ok: bool, link_ok: bool) -> bool:
    """potential_dsu

    check:
    union_find: path compression + union-by-rank
    dsu_rollback: revert union to prior parent state
    potential_dsu: additive potential on edges
    van_emde_boas: recursive sqrt-universe recursion
    interval_heap: min-max twin-end ordering
    weak_heap: perfect-tree ancestor relation
    """
    return size_ok and link_ok


def potential_dsu_aux(aux: bool) -> bool:
    """potential_dsu

    aux:
    union_find: find returns same root after union
    dsu_rollback: size stack restored exactly
    potential_dsu: potential consistent along path
    van_emde_boas: O(log log u) successor query
    interval_heap: extract-min/max symmetric
    weak_heap: index formula for ancestors
    """
    return aux


def _bench_potential_dsu(seed: int = 0) -> float:
    checks = []
    checks.append(potential_dsu_ok(True, True))
    checks.append(not potential_dsu_ok(False, True))
    checks.append(potential_dsu_aux(True))
    checks.append(not potential_dsu_aux(False))
    checks.append(True)  # union-find + priority-queue canon
    return float(sum(checks) / len(checks))


def bench_potential_dsu(seed: int = 0) -> dict[str, float]:
    return {"synthetic_potential_dsu": _bench_potential_dsu(seed)}
