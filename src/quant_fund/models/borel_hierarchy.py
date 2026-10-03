"""Borel hierarchy on finite topologies: Σ/Π rank via set operations (SYNTHETIC)."""

from __future__ import annotations


def complements(
    pts: frozenset[int], family: frozenset[frozenset[int]]
) -> frozenset[frozenset[int]]:
    return frozenset(pts - s for s in family)


def unions_all(family: frozenset[frozenset[int]]) -> frozenset[frozenset[int]]:
    """All unions of finite families (powerset unions — exhaustive on finite)."""
    out: set[frozenset[int]] = {frozenset()}
    for s in family:
        out |= {o | s for o in list(out)}
    return frozenset(out)


def intersections_all(family: frozenset[frozenset[int]]) -> frozenset[frozenset[int]]:
    fs = list(family)
    if not fs:
        return frozenset()
    acc = {fs[0]}
    for s in fs:
        acc = {a & s for a in acc} | {s}
    for s in fs:
        acc |= {s}
    return frozenset(acc)


def sigma_rank(
    pts: frozenset[int], opens: frozenset[frozenset[int]], target: frozenset[int]
) -> int:
    """Smallest Borel rank: 0=open, 1=F_sigma of open complements (closed sets union),
    2=G_delta_sigma, etc."""
    cur = opens
    for r in range(5):
        if target in cur:
            return r
        closed = complements(pts, cur)
        cur = unions_all(closed)
        if target in cur:
            return r * 2 + 1  # F_sigma level
        cur = intersections_all(closed)
        if target in cur:
            return r * 2 + 2
    return -1


def _bench_borel_hierarchy(seed: int = 0) -> float:
    checks = []
    pts = frozenset({0, 1, 2})
    opens = frozenset({frozenset(), frozenset({0}), frozenset({0, 1}), pts})
    checks.append(sigma_rank(pts, opens, frozenset({0})) == 0)
    # {2} is closed (complement of {0,1} open): rank F_sigma -> closed sets are Π1... sigma_rank via unions of closed: rank 1
    checks.append(sigma_rank(pts, opens, frozenset({2})) == 1)
    # unions closure
    checks.append(frozenset({0, 2}) in unions_all(frozenset({frozenset({0}), frozenset({2})})))
    # complement of open is closed
    checks.append(frozenset({1, 2}) in complements(pts, opens))
    # discrete: every set rank 0
    disc = frozenset(frozenset(s) for s in unions_all(frozenset(frozenset({x}) for x in pts)))
    checks.append(sigma_rank(pts, disc, frozenset({1, 2})) == 0)
    return float(sum(checks) / len(checks))


def bench_borel_hierarchy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_borel_hierarchy": _bench_borel_hierarchy(seed)}
