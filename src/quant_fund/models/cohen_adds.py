"""Cohen generic produces a new real (SYNTHETIC)."""

from __future__ import annotations

Cond = dict[int, int]


def generic_union(g: list[Cond]) -> dict[int, int]:
    """Union of a chain of conditions = the generic's induced function."""
    out: dict[int, int] = {}
    for c in g:
        for k, v in c.items():
            if k in out and out[k] != v:
                raise ValueError("incompatible conditions in generic")
            out[k] = v
    return out


def extends_ground(f_union: dict[int, int], ground: dict[int, int]) -> bool:
    """Does the generic function equal a ground-model function on dom(f)?"""
    return all(f_union.get(k) == v for k, v in ground.items()) and all(k in ground for k in f_union)


def _bench_cohen_adds(seed: int = 0) -> float:
    checks = []
    g = [{}, {0: 1}, {0: 1, 1: 0}, {0: 1, 1: 0, 2: 1}]
    u = generic_union(g)
    checks.append(u == {0: 1, 1: 0, 2: 1})
    # generic differs from every finite ground function it meets:
    # ground g0: n->0 for all n — u disagrees at 0
    ground0 = {0: 0, 1: 0, 2: 0}
    checks.append(not extends_ground(u, ground0))
    # ground g1: n->1 — disagrees at 1
    ground1 = {0: 1, 1: 1, 2: 1}
    checks.append(not extends_ground(u, ground1))
    # but u extends its own stem {0:1}: consistent on shared domain
    checks.append(all(u[k] == v for k, v in {0: 1}.items()))
    # incompatible generics raise
    try:
        generic_union([{0: 1}, {0: 0}])
        bad = False
    except ValueError:
        bad = True
    checks.append(bad)
    return float(sum(checks) / len(checks))


def bench_cohen_adds(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cohen_adds": _bench_cohen_adds(seed)}
