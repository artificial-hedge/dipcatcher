"""Knaster-Tarski least fixed point on a powerset lattice (SYNTHETIC)."""

from __future__ import annotations


def lfp(universe: frozenset[int], f, max_iter: int = 100) -> frozenset[int]:
    """Iterate f from bottom until fixed point (monotone f assumed)."""
    cur: frozenset[int] = frozenset()
    for _ in range(max_iter):
        nxt = frozenset(f(cur))
        if nxt == cur:
            return cur
        if not nxt <= universe:
            raise ValueError("f escapes universe")
        cur = nxt
    return cur


def gfp(universe: frozenset[int], f, max_iter: int = 100) -> frozenset[int]:
    cur: frozenset[int] = universe
    for _ in range(max_iter):
        nxt = frozenset(f(cur))
        if nxt == cur:
            return cur
        cur = nxt
    return cur


def reachable(adj: dict[int, set[int]], sources: frozenset[int]) -> frozenset[int]:
    """Standard lfp example: nodes reachable from sources."""
    return lfp(frozenset(adj), lambda s: sources | {v for u in s for v in adj.get(u, set())})


def _bench_tarski_fixed(seed: int = 0) -> float:
    checks = []
    adj = {0: {1}, 1: {2}, 2: set(), 3: {3}}
    checks.append(reachable(adj, frozenset({0})) == frozenset({0, 1, 2}))
    checks.append(reachable(adj, frozenset({3})) == frozenset({3}))

    # closure property: f(lfp) == lfp
    def f(s: frozenset[int]) -> frozenset[int]:
        return frozenset({0}) | {v for u in s for v in adj.get(u, set())}

    checks.append(f(reachable(adj, frozenset({0}))) == frozenset({0, 1, 2}))
    # gfp example: greatest invariant = nodes that don't reach a dead state? use
    # even-iteration parity set
    checks.append(gfp(frozenset({0, 1, 2}), lambda s: s | frozenset({0})) == frozenset({0, 1, 2}))
    return float(sum(checks) / len(checks))


def bench_tarski_fixed(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tarski_fixed": _bench_tarski_fixed(seed)}
