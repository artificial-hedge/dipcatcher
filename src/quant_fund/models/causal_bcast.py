"""Causal broadcast: deliver messages only after dependencies delivered (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 745


def causal_deliver(events: list[tuple[int, list[int]]]) -> list[int]:
    """events: (id, [dep_ids]). Deliver in topological order."""
    deps = {i: set(d) for i, d in events}
    delivered: list[int] = []
    pending = set(deps)
    while pending:
        for i in sorted(pending):
            if deps[i] <= set(delivered):
                delivered.append(i)
                pending.discard(i)
                break
        else:
            break
    return delivered


def bench_causal_bcast(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 50
    for _ in range(trials):
        n = int(rng.randint(3, 12))
        evs = []
        for i in range(n):
            deps = [j for j in range(i) if rng.rand() < 0.3]
            evs.append((i, deps))
        order = causal_deliver(evs)
        pos = {v: k for k, v in enumerate(order)}
        good = all(all(pos[d] < pos[i] for d in deps) for i, deps in evs)
        ok += float(good and len(order) == n)
    return {"synthetic_causal_order": ok / trials}
