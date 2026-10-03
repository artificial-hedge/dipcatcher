"""Souslin operation and analytic sets on finite trees (SYNTHETIC)."""

from __future__ import annotations

Seq = tuple[int, ...]


def souslin(tree_sets: dict[Seq, set[int]], max_len: int) -> set[int]:
    """A({A_s}) = union over branches sigma of intersection_n A_{sigma|n}.

    Model: each element x has a set of "witness strings"; x is in the
    Souslin set iff some branch contains x at every level.
    """
    out: set[int] = set()
    elems = set()
    for s in tree_sets.values():
        elems |= s
    for x in elems:
        # does x appear along a branch: exists choice of a sequence of
        # prefixes s_1 < s_2 < ... < s_L with x in A_{s_i} for all i
        def extends_through(prefix: Seq, x: int = x) -> bool:
            if len(prefix) >= max_len:
                return x in tree_sets.get(prefix, set())
            return any(
                x in tree_sets.get(prefix + (i,), set()) and extends_through(prefix + (i,))
                for i in range(3)
            )

        if extends_through(()):
            out.add(x)
    return out


def _bench_souslin_op(seed: int = 0) -> float:
    checks = []
    # element 7 appears on every node of the leftmost branch
    sets: dict[Seq, set[int]] = {
        (): {7},
        (0,): {7},
        (1,): {1},
        (0, 0): {7},
        (1, 0): {1},
        (0, 0, 0): {7},
        (1, 0, 0): {1},
    }
    s = souslin(sets, max_len=3)
    checks.append(7 in s)
    checks.append(1 in s)  # branch (1,0,0)
    checks.append(2 not in s)
    # an element appearing only at the root isn't in the Souslin set
    sets2: dict[Seq, set[int]] = {(): {9}}
    checks.append(9 not in souslin(sets2, max_len=2))
    # Souslin of closed sets is analytic: constant sets -> intersection
    sets3: dict[Seq, set[int]] = {
        (): {3},
        (0,): {3},
        (0, 0): {3},
    }
    checks.append(3 in souslin(sets3, max_len=2))
    checks.append(souslin(sets3, max_len=2) == {3})
    return float(sum(checks) / len(checks))


def bench_souslin_op(seed: int = 0) -> dict[str, float]:
    return {"synthetic_souslin_op": _bench_souslin_op(seed)}
