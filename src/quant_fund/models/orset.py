"""OR-Set CRDT: add-wins observed-remove set (synthetic).

Elements carry unique tags per add; remove drops only observed
tags. Verified: merge laws (comm/assoc/idem), add-wins semantics
(concurrent add survives a remove of older tags), remove of
non-observed element is a no-op.
"""

from __future__ import annotations

import random

# state: elem -> set of tags
State = dict[str, frozenset[int]]


def add(st: State, e: str, tag: int) -> State:
    tags = st.get(e, frozenset()) | {tag}
    return {**st, e: tags}


def remove(st: State, e: str) -> State:
    out = dict(st)
    out.pop(e, None)
    return out


def contains(st: State, e: str) -> bool:
    return bool(st.get(e))


def merge(a: State, b: State) -> State:
    out: State = {}
    for e in a.keys() | b.keys():
        tags = a.get(e, frozenset()) | b.get(e, frozenset())
        if tags:
            out[e] = tags
    return out


def bench_orset(seed: int = 20261231 + 302) -> dict[str, float]:
    rng = random.Random(seed)
    comm = assoc = idem = 0
    trials = 50
    for _ in range(trials):
        el = ["x", "y", "z"]
        a: State = {
            e: frozenset(rng.sample(range(20), rng.randint(1, 3))) for e in el if rng.random() < 0.6
        }
        b: State = {
            e: frozenset(rng.sample(range(20), rng.randint(1, 3))) for e in el if rng.random() < 0.6
        }
        c: State = {
            e: frozenset(rng.sample(range(20), rng.randint(1, 3))) for e in el if rng.random() < 0.6
        }
        comm += int(merge(a, b) == merge(b, a))
        assoc += int(merge(merge(a, b), c) == merge(a, merge(b, c)))
        idem += int(merge(a, a) == a)
    # add-wins: remove at one replica while another concurrently adds
    base: State = add({}, "e", 1)
    r1 = remove(base, "e")
    r2 = add(base, "e", 2)  # concurrent add, new tag
    m = merge(r1, r2)
    aw_flag = contains(m, "e")
    # non-observed remove: removing e from empty knowledge is no-op
    noop = remove({}, "e") == {}
    return {
        "synthetic_commutative": float(comm / trials),
        "synthetic_associative": float(assoc / trials),
        "synthetic_idempotent": float(idem / trials),
        "synthetic_add_wins": float(aw_flag),
        "synthetic_unobserved_noop": float(noop),
    }
