"""Conjugacy classes + class equation on finite groups (SYNTHIC)."""

from __future__ import annotations


def conjugate(mul, inv, g, x):
    # elements stay in the group's own representation — permutation
    # tuples for S3, plain ints for Z4; ``mul`` already returns it.
    return mul(mul(g, x), inv(g))


def class_of(g_elems: frozenset, mul, inv, x) -> frozenset:
    return frozenset(conjugate(mul, inv, g, x) for g in g_elems)


def all_classes(g_elems: frozenset, mul, inv) -> list[frozenset]:
    seen: set = set()
    out = []
    for x in g_elems:
        if x not in seen:
            c = class_of(g_elems, mul, inv, x)
            out.append(c)
            seen |= c
    return out


def centralizer(g_elems: frozenset, mul, x) -> frozenset:
    return frozenset(g for g in g_elems if mul(g, x) == mul(x, g))


def _bench_conjugacy_classes(seed: int = 0) -> float:
    checks = []
    s3 = frozenset({(0, 1, 2), (1, 0, 2), (0, 2, 1), (2, 1, 0), (1, 2, 0), (2, 0, 1)})

    def mul(p, q):
        return tuple(p[q[i]] for i in range(3))

    def inv(p):
        out = [0, 0, 0]
        for i, v in enumerate(p):
            out[v] = i
        return tuple(out)

    classes = all_classes(s3, mul, inv)
    checks.append(len(classes) == 3)  # e, transpositions, 3-cycles
    sizes = sorted(len(c) for c in classes)
    checks.append(sizes == [1, 2, 3])
    # class equation: sum = |G|
    checks.append(sum(len(c) for c in classes) == 6)
    # centralizer of transposition has order 2 -> class size 3 = |G|/|C|
    cz = centralizer(s3, mul, (1, 0, 2))
    checks.append(len(cz) == 2 and len(class_of(s3, mul, inv, (1, 0, 2))) == 3)
    # abelian Z4: all classes singleton
    z4 = frozenset(range(4))

    def mul4(a, b):
        return (a + b) % 4

    def inv4(a):
        return (-a) % 4

    checks.append(len(all_classes(z4, mul4, inv4)) == 4)
    return float(sum(checks) / len(checks))


def bench_conjugacy_classes(seed: int = 0) -> dict[str, float]:
    return {"synthetic_conjugacy_classes": _bench_conjugacy_classes(seed)}
