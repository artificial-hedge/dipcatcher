"""Ramsey/van der Waerden bounds via exhaustive coloring search (SYNTHETIC bench)."""

from __future__ import annotations

import itertools


def has_mono_clique(edges_color: dict[tuple[int, int], int], k: int, n: int) -> bool:
    """Some k-subset whose edges all share one color."""
    for sub in itertools.combinations(range(n), k):
        cols = {edges_color[(min(x, y), max(x, y))] for x, y in itertools.combinations(sub, 2)}
        if len(cols) == 1:
            return True
    return False


def r_bound(k: int, n: int, colors: int = 2) -> bool:
    """True iff EVERY r-coloring of K_n has a mono k-clique (n small)."""
    edge_list = [(x, y) for x in range(n) for y in range(x + 1, n)]
    for assign in itertools.product(range(colors), repeat=len(edge_list)):
        ec = dict(zip(edge_list, assign, strict=True))
        if not has_mono_clique(ec, k, n):
            return False
    return True


def vd_waerden(k: int, n: int, colors: int = 2) -> bool:
    """Every colors-coloring of [n] has a mono arithmetic progression of length k."""
    aps = [
        [a + i * d for i in range(k)]
        for a in range(1, n + 1)
        for d in range(1, n)
        if a + (k - 1) * d <= n
    ]
    for assign in itertools.product(range(colors), repeat=n):
        bad = False
        for ap in aps:
            if len({assign[x - 1] for x in ap}) == 1:
                bad = True
                break
        if not bad:
            return False
    return True


def _bench_ramsey_theory(seed: int = 0) -> float:
    checks = []
    # R(3,3)=6: K_5 has a 2-coloring without mono triangle; K_6 does not
    checks.append(not r_bound(3, 5))
    checks.append(r_bound(3, 6))
    # W(2,3)=9: [8] 2-colorable without mono 3-AP; [9] not
    checks.append(not vd_waerden(3, 8))
    checks.append(vd_waerden(3, 9))
    return sum(checks) / len(checks)


def bench_ramsey_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ramsey_theory": _bench_ramsey_theory(seed)}
