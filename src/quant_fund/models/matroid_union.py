"""Matroid union: independence = union of independents (SYNTHETIC)."""

from __future__ import annotations

from itertools import combinations


def union_independent(indeps: list[set[frozenset[int]]], n: int) -> set[frozenset[int]]:
    """Independent sets of M1 v M2 v ...: sets partitionable into
    independents of the summands."""
    all_sets = {frozenset(s) for r in range(n + 1) for s in combinations(range(n), r)}
    out: set[frozenset[int]] = set()
    for s in all_sets:
        elems = list(s)

        # can the elements be covered by independents of the summands?
        # recursive assignment
        def cover(
            idx: int,
            used: frozenset[int],
            parts: list[frozenset[int]],
            elems: tuple[int, ...] = tuple(elems),
        ) -> bool:
            if idx == len(elems):
                return all(p in indeps[i] for i, p in enumerate(parts))
            e = elems[idx]
            for i in range(len(indeps)):
                cand = frozenset(set(parts[i]) | {e})
                if cand in indeps[i]:
                    new_parts = list(parts)
                    new_parts[i] = cand
                    if cover(idx + 1, used | {e}, new_parts):
                        return True
            return False

        if cover(0, frozenset(), [frozenset()] * len(indeps)):
            out.add(s)
    return out


def _bench_matroid_union(seed: int = 0) -> float:
    checks = []

    # union of two partition matroids on 4 elements split {0,1}|{2,3}
    # each allowing one per part: M1=M2 permits sets covered by two picks
    def partition_indep(parts: list[set[int]], n: int) -> set[frozenset[int]]:
        out: set[frozenset[int]] = set()
        for r in range(n + 1):
            for s in combinations(range(n), r):
                fs = frozenset(s)
                if all(len(fs & p) <= 1 for p in parts):
                    out.add(fs)
        return out

    m1 = partition_indep([{0, 1}, {2, 3}], 4)
    m2 = partition_indep([{0, 2}, {1, 3}], 4)
    u = union_independent([m1, m2], 4)
    # {0,1,2,3}: split {0,3} in m1? {0,3}: one from each part of m1 ok;
    # {1,2} in m2: ok -> full set is in the union
    checks.append(frozenset({0, 1, 2, 3}) in u)
    # {0,1} needs two elements both in part {0,1} of m1 -> assign 0->m1,
    # 1->m2 (different parts in m2: 0 in {0,2}, 1 in {1,3}) ok
    checks.append(frozenset({0, 1}) in u)
    # union of a matroid with itself doubles rank: U_{1,4} v U_{1,4} = U_{2,4}
    u14 = {frozenset(s) for r in range(2) for s in combinations(range(4), r)}
    uu = union_independent([u14, u14], 4)
    checks.append(all(frozenset(s) in uu for s in combinations(range(4), 2)))
    checks.append(frozenset({0, 1, 2}) not in uu)
    return float(sum(checks) / len(checks))


def bench_matroid_union(seed: int = 0) -> dict[str, float]:
    return {"synthetic_matroid_union": _bench_matroid_union(seed)}
