"""WAL + ARIES-lite crash recovery — SYNTHETIC.

Log records (op, key, val, committed). Recovery replays committed
ops, discards uncommitted tail. Verified: recovered state ==
committed-prefix state; uncommitted writes invisible.
"""

from __future__ import annotations

import random

# log rec: ("begin",tx) ("set",tx,k,v) ("commit",tx) ("abort",tx)
Rec = tuple


def apply_wal(log: list[Rec]) -> dict[str, int]:
    db: dict[str, int] = {}
    staged: dict[int, dict[str, int]] = {}
    committed: set[int] = set()
    for r in log:
        if r[0] == "begin":
            staged[r[1]] = {}
        elif r[0] == "set":
            staged.setdefault(r[1], {})[r[2]] = r[3]
        elif r[0] == "commit":
            committed.add(r[1])
            for k, v in staged.pop(r[1], {}).items():
                db[k] = v
        elif r[0] == "abort":
            staged.pop(r[1], None)
    # crash: uncommitted staged writes must NOT appear
    return db


def bench_wal_recovery(seed: int = 20261231 + 361) -> dict[str, float]:
    rng = random.Random(seed)
    redo_ok = undo_ok = 0
    trials = 40
    for _ in range(trials):
        log: list[Rec] = []
        committed_expect: dict[str, int] = {}
        for tx in range(rng.randrange(3, 10)):
            log.append(("begin", tx))
            for _ in range(rng.randrange(1, 4)):
                k = f"k{rng.randrange(0, 5)}"
                v = rng.randrange(100)
                log.append(("set", tx, k, v))
                # track if this tx will commit
            if rng.random() < 0.7:
                log.append(("commit", tx))
                for r in log:
                    if r[0] == "set" and r[1] == tx:
                        committed_expect[r[2]] = r[3]
            else:
                pass  # leave uncommitted (crash)
        db = apply_wal(log)
        redo_ok += int(all(db[k] == v for k, v in committed_expect.items()))
        undo_ok += int(len(db) <= len(committed_expect))
    return {
        "synthetic_committed_visible": float(redo_ok / trials),
        "synthetic_uncommitted_gone": float(undo_ok / trials),
    }
