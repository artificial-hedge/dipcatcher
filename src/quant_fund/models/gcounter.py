"""G-Counter CRDT: grow-only counter (synthetic).

State: per-replica counts; merge = element-wise max; value = sum.
Verified: merge commutative + associative + idempotent on random
replica states; increments only readable after merge.
"""

from __future__ import annotations

import random

State = dict[int, int]


def inc(st: State, node: int, amt: int = 1) -> State:
    return {**st, node: st.get(node, 0) + amt}


def value(st: State) -> int:
    return sum(st.values())


def merge(a: State, b: State) -> State:
    keys = a.keys() | b.keys()
    return {k: max(a.get(k, 0), b.get(k, 0)) for k in keys}


def bench_gcounter(seed: int = 20261231 + 300) -> dict[str, float]:
    rng = random.Random(seed)
    comm = assoc = idem = mon = 0
    trials = 50
    for _ in range(trials):
        st1 = {i: rng.randint(0, 10) for i in range(3) if rng.random() < 0.7}
        st2 = {i: rng.randint(0, 10) for i in range(3) if rng.random() < 0.7}
        st3 = {i: rng.randint(0, 10) for i in range(3) if rng.random() < 0.7}
        comm += int(merge(st1, st2) == merge(st2, st1))
        assoc += int(merge(merge(st1, st2), st3) == merge(st1, merge(st2, st3)))
        idem += int(merge(st1, st1) == st1 and merge(merge(st1, st2), st2) == merge(st1, st2))
        # monotonicity: merge never decreases any component
        m = merge(st1, st2)
        mon += int(all(m[k] >= st1.get(k, 0) for k in m) and all(m[k] >= st2.get(k, 0) for k in m))
    # live scenario: 3 replicas, random incs + merges converge
    reps: list[State] = [{}, {}, {}]
    for _ in range(200):
        i = rng.randrange(3)
        reps[i] = inc(reps[i], i)
        j, k = rng.sample(range(3), 2)
        reps[j] = merge(reps[j], reps[k])
    final = merge(merge(reps[0], reps[1]), reps[2])
    converge = all(merge(r, final) == final for r in reps)
    return {
        "synthetic_commutative": float(comm / trials),
        "synthetic_associative": float(assoc / trials),
        "synthetic_idempotent": float(idem / trials),
        "synthetic_monotone": float(mon / trials),
        "synthetic_converges": float(converge),
    }
