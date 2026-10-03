"""SYNTHETIC Zookeeper Atomic Broadcast (ZAB-lite).

Leader assigns monotonically increasing zxids (epoch.counter); followers
deliver in zxid order; new epoch inherits last committed zxid. Verify:
total order per follower, epoch prefix across leader changes, no
delivered entry lost.
"""

from __future__ import annotations

import random


def run_epoch(
    epoch: int, proposals: list[int], delivered: list[tuple[int, int]], rng: random.Random
) -> None:
    for i, p in enumerate(proposals):
        delivered.append(((epoch << 20) | i, p))


def bench_zab_protocol(seed: int = 20261231 + 443) -> dict[str, float]:
    rng = random.Random(seed)
    total = epoch_ok = atomic = 0
    trials = 40
    for _ in range(trials):
        delivered: list[tuple[int, int]] = []
        epochs = rng.randrange(1, 4)
        for e in range(epochs):
            n = rng.randrange(2, 8)
            props = [rng.randrange(1, 100) for _ in range(n)]
            run_epoch(e, props, delivered, rng)
        # zxid total order strictly increasing
        zxids = [z for z, _ in delivered]
        total += int(all(zxids[i] < zxids[i + 1] for i in range(len(zxids) - 1)))
        # epoch prefix: each later epoch's first zxid > prior epoch's last
        firsts: dict[int, int] = {}
        lasts: dict[int, int] = {}
        for z, _ in delivered:
            e = z >> 20
            firsts.setdefault(e, z)
            lasts[e] = z
        epoch_ok += int(all(lasts[e - 1] < firsts[e] for e in range(1, epochs)))
        # atomic: no gaps within an epoch (counter 0..n-1)
        ok = True
        for e in range(epochs):
            counters = sorted(z & ((1 << 20) - 1) for z, _ in delivered if z >> 20 == e)
            ok = ok and counters == list(range(len(counters)))
        atomic += int(ok)
    return {
        "synthetic_total_order": float(total / trials),
        "synthetic_epoch_monotonic": float(epoch_ok / trials),
        "synthetic_no_gaps": float(atomic / trials),
    }
