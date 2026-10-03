"""Planar rooted-tree operad: grafting composition (SYNTHETIC)."""

from __future__ import annotations

from math import comb


def planar_binary_trees(n_leaves: int) -> int:
    """Number of planar binary trees with n leaves = Catalan(n-1)."""
    return comb(2 * (n_leaves - 1), n_leaves - 1) // n_leaves


def graft(outer: tuple, slot: int, inner: tuple) -> tuple:
    """Graft `inner` tree onto the `slot`-th leaf of `outer`.

    Trees are nested tuples: a leaf is (); an internal node is a tuple of
    subtrees. slot indexes leaves in planar order.
    """
    leaves = [0]

    def rec(t: tuple) -> tuple:
        if t == ():
            if leaves[0] == slot:
                leaves[0] += 1
                return inner
            leaves[0] += 1
            return t
        return tuple(rec(s) for s in t)

    return rec(outer)


def leaf_count(t: tuple) -> int:
    if t == ():
        return 1
    return sum(leaf_count(s) for s in t)


def _bench_operad_tree(seed: int = 0) -> float:
    checks = []
    # Catalan counts
    checks.append(planar_binary_trees(4) == 5)
    checks.append(planar_binary_trees(3) == 2)
    checks.append(planar_binary_trees(5) == 14)
    # graft on the comb tree ((),()) slot 1 with ((),())
    t = graft(((), ()), 1, ((), ()))
    checks.append(t == ((), ((), ())))
    checks.append(leaf_count(t) == 3)
    # graft associativity: graft at slot j after grafting at slot i
    a = graft(((), ()), 0, ((), ()))
    b = graft(a, 1, ((), ()))
    c = graft(((), ()), 0, graft(((), ()), 1, ((), ())))
    checks.append(b == c)
    # grafting at slot 0 vs slot 2 give different trees (planarity matters)
    checks.append(graft(((), ()), 0, ((), ())) != graft(((), ()), 1, ((), ())))
    return float(sum(checks) / len(checks))


def bench_operad_tree(seed: int = 0) -> dict[str, float]:
    return {"synthetic_operad_tree": _bench_operad_tree(seed)}
