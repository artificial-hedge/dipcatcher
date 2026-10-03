"""PN-Counter CRDT: increment/decrement counter (synthetic).

Two G-Counters (P and N); value = P.sum − N.sum; merge = pairwise
max per counter. Verified: merge laws hold; dec-applied value
equals net increments.
"""

from __future__ import annotations

import random

State = tuple[dict[int, int], dict[int, int]]


def inc(st: State, node: int) -> State:
    p, n = st
    return {**p, node: p.get(node, 0) + 1}, dict(n)


def dec(st: State, node: int) -> State:
    p, n = st
    return dict(p), {**n, node: n.get(node, 0) + 1}


def value(st: State) -> int:
    return sum(st[0].values()) - sum(st[1].values())


def merge(a: State, b: State) -> State:
    def mx(x: dict[int, int], y: dict[int, int]) -> dict[int, int]:
        return {k: max(x.get(k, 0), y.get(k, 0)) for k in x.keys() | y.keys()}

    return mx(a[0], b[0]), mx(a[1], b[1])


def bench_pncounter(seed: int = 20261231 + 301) -> dict[str, float]:
    rng = random.Random(seed)
    comm = assoc = idem = 0
    trials = 50
    for _ in range(trials):
        a: State = (
            {i: rng.randint(0, 5) for i in range(3) if rng.random() < 0.6},
            {i: rng.randint(0, 5) for i in range(3) if rng.random() < 0.6},
        )
        b: State = (
            {i: rng.randint(0, 5) for i in range(3) if rng.random() < 0.6},
            {i: rng.randint(0, 5) for i in range(3) if rng.random() < 0.6},
        )
        c: State = (
            {i: rng.randint(0, 5) for i in range(3) if rng.random() < 0.6},
            {i: rng.randint(0, 5) for i in range(3) if rng.random() < 0.6},
        )
        comm += int(merge(a, b) == merge(b, a))
        assoc += int(merge(merge(a, b), c) == merge(a, merge(b, c)))
        idem += int(merge(a, a) == a)
    # semantics: inc/dec net equals value after full merge
    reps: list[State] = [({}, {}), ({}, {})]
    ops = 0
    for _ in range(100):
        i = rng.randrange(2)
        if rng.random() < 0.5:
            reps[i] = inc(reps[i], i)
            ops += 1
        else:
            reps[i] = dec(reps[i], i)
            ops -= 1
        reps[0] = merge(reps[0], reps[1])
        reps[1] = merge(reps[1], reps[0])
    sem_ok = value(merge(reps[0], reps[1])) == ops
    return {
        "synthetic_commutative": float(comm / trials),
        "synthetic_associative": float(assoc / trials),
        "synthetic_idempotent": float(idem / trials),
        "synthetic_net_value": float(sem_ok),
    }
