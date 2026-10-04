"""Congruences on finite groups: normal subgroups and their lattice (SYNTHETIC)."""

from __future__ import annotations


def subgroup_closure(elems: frozenset[int], mul, gen: frozenset[int]) -> frozenset[int]:
    cur = set(gen)
    changed = True
    while changed:
        changed = False
        for a in list(cur):
            for b in list(cur):
                if mul(a, b) not in cur:
                    cur.add(mul(a, b))
                    changed = True
    return frozenset(cur)


def is_normal(g_elems: frozenset, mul, inv, h: frozenset) -> bool:
    for g in g_elems:
        for x in h:
            if mul(mul(g, x), inv(g)) not in h:
                return False
    return True


def subgroups(g_elems: frozenset[int], mul, e: int) -> frozenset[frozenset[int]]:
    from itertools import combinations

    out = set()
    elems = sorted(g_elems)
    for r in range(len(elems) + 1):
        for c in combinations(elems, r):
            s = frozenset(c)
            if e not in s:
                continue
            if subgroup_closure(g_elems, mul, s) == s:
                out.add(s)
    return frozenset(out)


def _bench_congruence_lattice(seed: int = 0) -> float:
    checks = []
    # Z4 additive
    elems = frozenset({0, 1, 2, 3})

    def mul(a: int, b: int) -> int:
        return (a + b) % 4

    def inv(a: int) -> int:
        return (-a) % 4

    subs = subgroups(elems, mul, 0)
    checks.append(frozenset({0, 2}) in subs)
    checks.append(frozenset({0, 1, 2, 3}) in subs)
    checks.append(len(subs) == 3)  # {0}, {0,2}, Z4
    # abelian => every subgroup normal
    checks.append(all(is_normal(elems, mul, inv, h) for h in subs))
    # S3: <(12)> not normal
    s3 = {(0, 1, 2), (1, 0, 2), (0, 2, 1), (2, 1, 0), (1, 2, 0), (2, 0, 1)}

    def comp(p, q):
        return tuple(p[q[i]] for i in range(3))

    def invp(p):
        out = [0, 0, 0]
        for i, v in enumerate(p):
            out[v] = i
        return tuple(out)

    s3e = frozenset(s3)
    h12 = frozenset({(0, 1, 2), (1, 0, 2)})  # {e, swap 0<->1}
    checks.append(not is_normal(s3e, comp, invp, h12))
    a3 = frozenset({(0, 1, 2), (1, 2, 0), (2, 0, 1)})
    checks.append(is_normal(s3e, comp, invp, a3))
    return float(sum(checks) / len(checks))


def bench_congruence_lattice(seed: int = 0) -> dict[str, float]:
    return {"synthetic_congruence_lattice": _bench_congruence_lattice(seed)}
