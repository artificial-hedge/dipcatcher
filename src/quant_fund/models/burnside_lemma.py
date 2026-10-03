"""Burnside lemma: orbit counting under group action (SYNTHETIC)."""

from __future__ import annotations


def orbit_count(g_elems: frozenset, act, elems: frozenset) -> int:
    """#orbits = (1/|G|) * sum_g |Fix(g)|."""
    fix_sum = 0
    for g in g_elems:
        for x in elems:
            if act(g, x) == x:
                fix_sum += 1
    assert fix_sum % len(g_elems) == 0
    return fix_sum // len(g_elems)


def direct_orbits(act, g_elems: frozenset, elems: frozenset) -> int:
    """Brute-force orbit partition for cross-check."""
    seen: set = set()
    orbits = 0
    for x in elems:
        if x in seen:
            continue
        orbits += 1
        seen |= {act(g, x) for g in g_elems}
    return orbits


def _bench_burnside_lemma(seed: int = 0) -> float:
    checks = []
    # Z2 acting on 3 binary strings by complement: elems = {0,1}^3
    elems = frozenset((a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1))
    z2 = frozenset({0, 1})

    def act(g, x):
        return x if g == 0 else tuple(1 - v for v in x)

    checks.append(orbit_count(z2, act, elems) == 4)
    checks.append(direct_orbits(act, z2, elems) == 4)
    # rotations of square vertices: Z4 on 4 elements -> 1 orbit
    z4 = frozenset(range(4))
    elems4 = frozenset(range(4))

    def act4(g, x):
        return (x + g) % 4

    checks.append(orbit_count(z4, act4, elems4) == 1)
    # 2-colorings of 4 beads necklace under Z4: known count 6
    beads = frozenset(tuple((x >> i) & 1 for i in range(4)) for x in range(16))

    def rot(g, x):
        return tuple(x[(i + g) % 4] for i in range(4))

    checks.append(orbit_count(z4, rot, beads) == 6)
    checks.append(direct_orbits(rot, z4, beads) == 6)
    return float(sum(checks) / len(checks))


def bench_burnside_lemma(seed: int = 0) -> dict[str, float]:
    return {"synthetic_burnside_lemma": _bench_burnside_lemma(seed)}
