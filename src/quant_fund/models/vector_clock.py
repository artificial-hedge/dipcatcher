"""Vector clocks for causality tracking (synthetic) (SYNTHETIC).

Implements increment/merge/compare on vector clocks over a random
event graph with concurrent and causal events. Verified: (i) a
happened-before relation computed by VC comparison agrees with
transitive-closure oracle on the event DAG; (ii) concurrent events
compare incomparable; (iii) merge is idempotent/commutative.
"""

from __future__ import annotations

import random

VC = tuple[int, ...]


def _le(a: VC, b: VC) -> bool:
    return all(x <= y for x, y in zip(a, b, strict=True))


def compare(a: VC, b: VC) -> str:
    if a == b:
        return "eq"
    if _le(a, b):
        return "before"
    if _le(b, a):
        return "after"
    return "concurrent"


def merge(a: VC, b: VC) -> VC:
    return tuple(max(x, y) for x, y in zip(a, b, strict=True))


def bench_vector_clock(seed: int = 20261231 + 252) -> dict[str, float]:
    rng = random.Random(seed)
    n_proc = 3
    # build a random event DAG: local events + sends
    n_events = 40
    events: list[tuple[int, list[int]]] = []  # (proc, parents as event idx)
    clocks: list[VC] = []
    for _ in range(n_events):
        p = rng.randrange(n_proc)
        local_prev = [i for i, e in enumerate(events) if e[0] == p]
        parents = [local_prev[-1]] if local_prev else []
        if rng.random() < 0.3 and events:
            parents.append(rng.randrange(len(events)))
        events.append((p, parents))
        vc = list(clocks[parents[0]]) if parents else [0] * n_proc
        for pa in parents[1:]:
            vc = list(merge(tuple(vc), clocks[pa]))
        vc[p] += 1
        clocks.append(tuple(vc))
    # oracle: transitive closure over parent edges
    reach: list[set[int]] = [set() for _ in events]
    for i, (_, parents) in enumerate(events):
        for pa in parents:
            reach[i] |= {pa} | reach[pa]
    agree = 0
    total = 0
    conc_ok = 0
    conc_total = 0
    for i in range(n_events):
        for j in range(i + 1, n_events):
            oracle = "before" if i in reach[j] else ("after" if j in reach[i] else "concurrent")
            vc_cmp = compare(clocks[i], clocks[j])
            agree += int(vc_cmp == oracle)
            total += 1
            if oracle == "concurrent":
                conc_total += 1
                conc_ok += int(vc_cmp == "concurrent")
    idem = all(merge(c, c) == c for c in clocks)
    comm = all(
        merge(clocks[i], clocks[j]) == merge(clocks[j], clocks[i])
        for i in range(10)
        for j in range(10)
    )
    return {
        "synthetic_agree": float(agree / max(1, total)),
        "synthetic_concurrent": float(conc_ok / max(1, conc_total)),
        "synthetic_idempotent": float(idem),
        "synthetic_commutative": float(comm),
    }
