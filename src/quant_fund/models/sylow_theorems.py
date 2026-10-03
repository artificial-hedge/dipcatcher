"""Sylow theorems verification on finite groups (SYNTHETIC)."""

from __future__ import annotations


def order_of(g: tuple, mul, ident: tuple) -> int:
    cur, n = g, 1
    while cur != ident:
        cur = mul(cur, g)
        n += 1
        if n > 1000:
            return -1
    return n


def subgroup_generated(g_elems: frozenset, mul, gen: frozenset) -> frozenset:
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


def all_subgroups(g_elems: frozenset, mul, ident) -> frozenset[frozenset]:
    subs = set()
    for g in g_elems:
        subs.add(subgroup_generated(g_elems, mul, frozenset({ident, g})))
    for a in g_elems:
        for b in g_elems:
            subs.add(subgroup_generated(g_elems, mul, frozenset({ident, a, b})))
    return frozenset(s for s in subs)


def sylow_p_subgroups(g_elems: frozenset, mul, ident, p: int) -> frozenset[frozenset]:
    """Subgroups of order p^k where p^k is max p-power dividing |G|."""
    n = len(g_elems)
    pk = 1
    while n % (pk * p) == 0:
        pk *= p
    subs = all_subgroups(g_elems, mul, ident)
    return frozenset(s for s in subs if len(s) == pk)


def _bench_sylow_theorems(seed: int = 0) -> float:
    checks = []
    # S3 as permutations of {0,1,2}
    s3 = frozenset({(0, 1, 2), (1, 0, 2), (0, 2, 1), (2, 1, 0), (1, 2, 0), (2, 0, 1)})

    def mul(p, q):
        return tuple(p[q[i]] for i in range(3))

    ident = (0, 1, 2)
    syl2 = sylow_p_subgroups(s3, mul, ident, 2)
    syl3 = sylow_p_subgroups(s3, mul, ident, 3)
    checks.append(len(syl2) == 3)  # n_2 = 3
    checks.append(len(syl3) == 1)  # n_3 = 1
    checks.append(len(syl2) % 2 == 1 and 6 % len(syl2) == 0)
    checks.append(len(syl3) % 3 == 1 and 6 % len(syl3) == 0)
    # Z6: unique Sylows
    z6 = frozenset(range(6))

    def mul6(a, b):
        return (a + b) % 6

    syl2z = sylow_p_subgroups(z6, mul6, 0, 2)
    syl3z = sylow_p_subgroups(z6, mul6, 0, 3)
    checks.append(len(syl2z) == 1 and len(syl3z) == 1)
    return float(sum(checks) / len(checks))


def bench_sylow_theorems(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sylow_theorems": _bench_sylow_theorems(seed)}
