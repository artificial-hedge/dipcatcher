"""SYNTHETIC mark-sweep + copying GC on an object graph.

Objects = nodes with child edges; roots given. Verify: mark-sweep frees
exactly unreachable, copying collector compacts survivors contiguously,
double collection is stable.
"""

from __future__ import annotations

import random


def mark(adj: dict[int, list[int]], roots: list[int]) -> set[int]:
    live: set[int] = set()
    stack = list(roots)
    while stack:
        u = stack.pop()
        if u in live:
            continue
        live.add(u)
        stack.extend(adj.get(u, []))
    return live


def sweep(adj: dict[int, list[int]], live: set[int]) -> dict[int, list[int]]:
    return {k: [c for c in v if c in live] for k, v in adj.items() if k in live}


def copy_gc(adj: dict[int, list[int]], roots: list[int]) -> dict[int, list[int]]:
    """Cheney-style copying: survivors get compact fresh addresses."""
    live = mark(adj, roots)
    order = list(roots)
    seen = set(roots)
    i = 0
    while i < len(order):
        for c in adj.get(order[i], []):
            if c in live and c not in seen:
                seen.add(c)
                order.append(c)
        i += 1
    remap = {old: j for j, old in enumerate(order)}
    return {remap[k]: [remap[c] for c in adj[k] if c in remap] for k in order}


def bench_gc_marksweep(seed: int = 20261231 + 414) -> dict[str, float]:
    rng = random.Random(seed)
    free_exact = compact = stable = 0
    trials = 40
    for _ in range(trials):
        n = rng.randrange(5, 15)
        adj = {i: [j for j in range(n) if j != i and rng.random() < 0.2] for i in range(n)}
        roots = [0]
        live = mark(adj, roots)
        freed = set(adj) - live
        after = sweep(adj, live)
        free_exact += int(set(after) == live and freed == set(adj) - live)
        cp = copy_gc(adj, roots)
        compact += int(sorted(cp) == list(range(len(live))))
        # second collection on compacted graph is stable
        cp2 = copy_gc(cp, [0])
        stable += int(len(mark(cp, [0])) == len(live) and set(cp2) == set(cp))
    return {
        "synthetic_frees_unreachable": float(free_exact / trials),
        "synthetic_copy_compacts": float(compact / trials),
        "synthetic_double_gc_stable": float(stable / trials),
    }
