"""Galois connections between powerset lattices (SYNTHETIC)."""

from __future__ import annotations


def is_monotone_gc(
    le_a: dict[tuple[int, int], bool],
    le_b: dict[tuple[int, int], bool],
    f,
    g,
    a_elems: set[int],
    b_elems: set[int],
) -> bool:
    """Monotone GC: f(p) <= b iff p <= g(b)."""
    for p in a_elems:
        for b in b_elems:
            lhs = le_b.get((f(p), b), False)
            rhs = le_a.get((p, g(b)), False)
            if lhs != rhs:
                return False
    return True


def closure_under(le: dict[tuple[int, int], bool], gf, a_elems: set[int]) -> dict[int, int]:
    """Closure operator p -> g(f(p)): p <= gf(p), idempotent, monotone."""
    return {p: gf(p) for p in a_elems}


def _bench_galois_connection(seed: int = 0) -> float:
    checks = []
    # floor/ceil GC between Z and R modeled on int grid: f(n)=n, g(m)=floor(m)
    a_elems = {0, 1, 2, 3}
    b_elems = {0, 1, 2, 3, 4, 5}
    le_a = {(x, y): x <= y for x in a_elems for y in a_elems}
    le_b = {(x, y): x <= y for x in b_elems for y in b_elems}

    def f(p: int) -> int:
        return 2 * p

    def g(b: int) -> int:
        return b // 2

    # 2p <= b iff p <= b//2 -> GC holds
    checks.append(is_monotone_gc(le_a, le_b, f, g, a_elems, b_elems))
    # bad g
    checks.append(
        not is_monotone_gc(le_a, le_b, f, lambda b: b // 2 + (1 if b % 2 else 0), a_elems, b_elems)
    )
    # closure: p <= g(f(p))
    checks.append(all(g(f(p)) >= p for p in a_elems))
    checks.append(all(f(g(b)) <= b for b in b_elems))
    return float(sum(checks) / len(checks))


def bench_galois_connection(seed: int = 0) -> dict[str, float]:
    return {"synthetic_galois_connection": _bench_galois_connection(seed)}
