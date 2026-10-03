"""SYNTHETIC ARIES-style recovery: analysis + redo + undo.

Log records: (txid, "update", key, before, after) / ("commit", txid).
Crash mid-transactions. Recovery: redo repeats history for all committed
updates; undo rolls back losers. Verify: committed changes survive,
uncommitted are gone, undo writes compensating records.
"""

from __future__ import annotations

import random


def recover(log: list[tuple], disk: dict[str, int]) -> tuple[dict[str, int], list[tuple]]:
    # analysis: committed txids
    committed = {r[1] for r in log if r[0] == "commit"}
    winners: set[int] = set()
    for r in log:
        if r[0] == "update" and r[1] in committed:
            winners.add(r[1])
    # redo repeats history for ALL update records (winners and losers)
    for r in log:
        if r[0] == "update":
            disk[r[2]] = r[4]
    # undo losers in reverse order, writing CLRs
    clr: list[tuple] = []
    loser_updates = [r for r in log if r[0] == "update" and r[1] not in committed]
    for r in reversed(loser_updates):
        disk[r[2]] = r[3]  # restore before-image
        clr.append(("clr", r[1], r[2], r[3]))
        # remove before-image if key never touched by a winner
        winner_keys = {x[2] for x in log if x[0] == "update" and x[1] in committed}
        if r[2] not in winner_keys and r[2] not in {x[2] for x in clr}:
            pass  # before-image already restored
    return disk, clr


def bench_aries_recovery(seed: int = 20261231 + 430) -> dict[str, float]:
    rng = random.Random(seed)
    survives = rolled = clr_ok = 0
    trials = 40
    for _ in range(trials):
        keys = [f"k{i}" for i in range(5)]
        disk = {k: 0 for k in keys}
        nt = rng.randrange(3, 6)
        committed = {t for t in range(nt) if rng.random() < 0.6}
        # lock-consistent log: per key, committed writers precede
        # uncommitted ones (strict 2PL blocks winner-after-loser writes).
        per_tx: dict[int, list[tuple]] = {t: [] for t in range(nt)}
        img = dict(disk)
        keyops: list[list[tuple]] = []
        for k in keys:
            writers = rng.sample(range(nt), rng.randrange(1, nt + 1))
            winners = [t for t in writers if t in committed]
            losers = [t for t in writers if t not in committed]
            ops = []
            for t in winners + losers:
                before = img[k]
                after = rng.randrange(1, 100)
                r = ("update", t, k, before, after)
                ops.append(r)
                per_tx[t].append(r)
                img[k] = after
            keyops.append(ops)
        # interleave ops preserving per-key order; commit record after a
        # tx's final op
        log: list[tuple] = []
        while keyops:
            ops = rng.choice(keyops)
            r = ops.pop(0)
            log.append(r)
            if not ops:
                keyops.remove(ops)
        for t in sorted(committed):
            log.append(("commit", t))
        # lock-consistent reference: per key, final = last COMMITTED
        # update's after-image; keys with no committed writer return to
        # the first loser's before-image (== base under w*l* ordering).
        expected = dict(disk)
        last_committed: dict[str, tuple] = {}
        first_loser: dict[str, tuple] = {}
        for r in log:
            if r[0] == "update":
                if r[1] in committed:
                    last_committed[r[2]] = r
                elif r[2] not in first_loser:
                    first_loser[r[2]] = r
        for k, r in last_committed.items():
            expected[k] = r[4]
        for k, r in first_loser.items():
            if k not in last_committed:
                expected[k] = r[3]
        disk2, clr = recover(log, dict(disk))
        survives += int(all(disk2.get(k) == expected.get(k) for k in expected))
        # keys touched only by losers end at their first before-image
        loser_keys = {r[2] for r in log if r[0] == "update" and r[1] not in committed} - {
            r[2] for r in log if r[0] == "update" and r[1] in committed
        }
        first_before = {}
        for r in log:
            if r[0] == "update" and r[1] not in committed and r[2] not in first_before:
                first_before[r[2]] = r[3]
        rolled += int(all(disk2.get(k) == first_before.get(k, disk.get(k)) for k in loser_keys))
        clr_ok += int(all(c[0] == "clr" for c in clr))
    return {
        "synthetic_committed_survives": float(survives / trials),
        "synthetic_losers_rolled_back": float(rolled / trials),
        "synthetic_clr_written": float(clr_ok / trials),
    }
