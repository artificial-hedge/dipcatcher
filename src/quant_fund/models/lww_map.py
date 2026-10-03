"""LWW-Register / LWW-Map CRDT (synthetic).

Each entry carries (timestamp, replica, value); merge picks the
max (ts, replica) per key. Verified: merge laws; concurrent
updates resolve deterministically; last write observed wins.
"""

from __future__ import annotations

import random

Entry = tuple[int, int, str]
State = dict[str, Entry]


def assign(st: State, key: str, ts: int, node: int, val: str) -> State:
    cur = st.get(key)
    cand: Entry = (ts, node, val)
    if cur is None or (cur[0], cur[1], cur[2]) < (ts, node, val):
        return {**st, key: cand}
    return dict(st)


def get(st: State, key: str) -> str | None:
    e = st.get(key)
    return e[2] if e else None


def merge(a: State, b: State) -> State:
    out: State = {}
    for k in a.keys() | b.keys():
        ea, eb = a.get(k), b.get(k)
        if ea is None:
            out[k] = eb  # type: ignore[assignment]
        elif eb is None:
            out[k] = ea
        else:
            out[k] = max(ea, eb, key=lambda e: (e[0], e[1], e[2]))
    return out


def bench_lww_map(seed: int = 20261231 + 303) -> dict[str, float]:
    rng = random.Random(seed)
    comm = assoc = idem = lww = 0
    trials = 50
    for _ in range(trials):
        keys = ["a", "b", "c"]
        a: State = {
            k: (rng.randint(0, 10), rng.randint(0, 2), f"v{k}1") for k in keys if rng.random() < 0.6
        }
        b: State = {
            k: (rng.randint(0, 10), rng.randint(0, 2), f"v{k}2") for k in keys if rng.random() < 0.6
        }
        c: State = {
            k: (rng.randint(0, 10), rng.randint(0, 2), f"v{k}3") for k in keys if rng.random() < 0.6
        }
        comm += int(merge(a, b) == merge(b, a))
        assoc += int(merge(merge(a, b), c) == merge(a, merge(b, c)))
        idem += int(merge(a, a) == a)
        # LWW: for any shared key the max-ts entry wins
        ok = True
        for k in a.keys() & b.keys():
            ea, eb = a[k], b[k]
            win = max(ea, eb, key=lambda e: (e[0], e[1], e[2]))
            ok = ok and merge(a, b)[k] == win
        lww += int(ok)
    return {
        "synthetic_commutative": float(comm / trials),
        "synthetic_associative": float(assoc / trials),
        "synthetic_idempotent": float(idem / trials),
        "synthetic_lww_wins": float(lww / trials),
    }
