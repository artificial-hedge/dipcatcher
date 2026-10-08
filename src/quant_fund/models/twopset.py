"""Two-Phase Set CRDT (2P-Set): add + tombstone sets (synthetic) (SYNTHETIC).

Element in set iff in A and not in R; once removed, never returns.
Verified: merge laws; remove-wins semantics; re-add after remove
is ineffective (2P semantics).
"""

from __future__ import annotations

import random

State = tuple[frozenset[str], frozenset[str]]


def add(st: State, e: str) -> State:
    return st[0] | {e}, st[1]


def remove(st: State, e: str) -> State:
    return st[0], st[1] | {e}


def contains(st: State, e: str) -> bool:
    return e in st[0] and e not in st[1]


def merge(a: State, b: State) -> State:
    return a[0] | b[0], a[1] | b[1]


def bench_twopset(seed: int = 20261231 + 304) -> dict[str, float]:
    rng = random.Random(seed)
    comm = assoc = idem = rw = 0
    trials = 50
    el = ["x", "y", "z"]
    for _ in range(trials):
        a: State = (
            frozenset(e for e in el if rng.random() < 0.6),
            frozenset(e for e in el if rng.random() < 0.4),
        )
        b: State = (
            frozenset(e for e in el if rng.random() < 0.6),
            frozenset(e for e in el if rng.random() < 0.4),
        )
        c: State = (
            frozenset(e for e in el if rng.random() < 0.6),
            frozenset(e for e in el if rng.random() < 0.4),
        )
        comm += int(merge(a, b) == merge(b, a))
        assoc += int(merge(merge(a, b), c) == merge(a, merge(b, c)))
        idem += int(merge(a, a) == a)
        # remove-wins: element in both A-sets but R in one → absent
        e = "x"
        x1: State = (frozenset({e}), frozenset())
        x2: State = (frozenset({e}), frozenset({e}))
        rw += int(not contains(merge(x1, x2), e))
    # re-add after observed remove stays absent
    st = add((frozenset(), frozenset()), "k")
    st = remove(st, "k")
    st = add(st, "k")
    readd_ok = not contains(st, "k")
    return {
        "synthetic_commutative": float(comm / trials),
        "synthetic_associative": float(assoc / trials),
        "synthetic_idempotent": float(idem / trials),
        "synthetic_remove_wins": float(rw / trials),
        "synthetic_no_readd": float(readd_ok),
    }
