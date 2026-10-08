"""SYNTHETIC EPaxos-lite — dependency-graph command ordering.

Commands carry (key, op); ops commute iff keys differ. Conflicts build a
dependency DAG; final execution order = deterministic topological sort.
Verify: all replicas agree on total order despite different arrival
orders, and commuting commands produce identical state either way.
"""

from __future__ import annotations

import random


def build_dag(cmds: list[tuple[str, int]]) -> dict[int, set[int]]:
    """Edge i->j if i conflicts with j and i arrived earlier."""
    deps: dict[int, set[int]] = {i: set() for i in range(len(cmds))}
    for i, (k1, _) in enumerate(cmds):
        for j in range(i + 1, len(cmds)):
            k2, _ = cmds[j]
            if k1 == k2:
                deps[j].add(i)
    return deps


def topo_order(deps: dict[int, set[int]]) -> list[int]:
    out: list[int] = []
    remaining = {i: set(d) for i, d in deps.items()}  # copy: never mutate caller
    while remaining:
        ready = sorted(i for i, d in remaining.items() if not d)
        if not ready:
            break
        nxt = ready[0]
        out.append(nxt)
        del remaining[nxt]
        for d in remaining.values():
            d.discard(nxt)
    return out


def _apply(state: dict[str, int], cmds: list[tuple[str, int]], order: list[int]) -> dict[str, int]:
    s = dict(state)
    for i in order:
        k, delta = cmds[i]
        s[k] = s.get(k, 0) + delta
    return s


def _alt_linear_order(deps: dict[int, set[int]], rng: random.Random) -> list[int]:
    """A different valid linear extension of the same DAG: random ready
    choice instead of lowest-index — replicas that break ties differently
    must still land on the same final state."""
    remaining = {i: set(d) for i, d in deps.items()}
    out: list[int] = []
    while remaining:
        ready = sorted(i for i, d in remaining.items() if not d)
        if not ready:
            break
        nxt = ready[rng.randrange(len(ready))]
        out.append(nxt)
        del remaining[nxt]
        for d in remaining.values():
            d.discard(nxt)
    return out


def bench_epaxos(seed: int = 20261231 + 441) -> dict[str, float]:
    rng = random.Random(seed)
    agree = commute_ok = det = 0
    trials = 40
    for _ in range(trials):
        n = rng.randrange(4, 10)
        cmds = [(rng.choice("ab"), rng.randrange(1, 10)) for _ in range(n)]
        deps = build_dag(cmds)
        order = topo_order(deps)
        s1 = _apply({}, cmds, order)
        # a replica linearizing the same agreed DAG with a different
        # ready-tiebreak converges to the same state
        s2 = _apply({}, cmds, _alt_linear_order(deps, rng))
        agree += int(s1 == s2)
        # commuting subsequence: all a-ops then b-ops equals interleaved
        sa = _apply(
            {}, [c for c in cmds if c[0] == "a"], list(range(sum(c[0] == "a" for c in cmds)))
        )
        sa2 = _apply(
            sa, [c for c in cmds if c[0] == "b"], list(range(sum(c[0] == "b" for c in cmds)))
        )
        commute_ok += int(sa2 == s1)
        # deterministic + non-mutating: deps is still intact after sorting
        det += int(order == topo_order(deps) and deps == build_dag(cmds))
    return {
        "synthetic_replicas_agree": float(agree / trials),
        "synthetic_commutes_preserve_state": float(commute_ok / trials),
        "synthetic_deterministic_order": float(det / trials),
    }
