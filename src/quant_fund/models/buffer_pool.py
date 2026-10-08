"""SYNTHETIC buffer-pool with dirty-page write-back + clock eviction.

Pages loaded on demand; dirty pages flushed on eviction. Clock (second
chance) approximates LRU. Verify: writes survive eviction (flush-before-
evict), hot pages stay resident, clock beats FIFO on a skewed trace.
"""

from __future__ import annotations

import random


def fifo_run(trace: list[int], cap: int) -> int:
    buf: list[int] = []
    misses = 0
    for p in trace:
        if p not in buf:
            misses += 1
            if len(buf) >= cap:
                buf.pop(0)
            buf.append(p)
    return misses


def clock_run(trace: list[int], cap: int) -> int:
    buf: list[int] = []
    ref: dict[int, bool] = {}
    hand = 0
    misses = 0
    for p in trace:
        if p in buf:
            ref[p] = True
            continue
        misses += 1
        if len(buf) < cap:
            buf.append(p)
            ref[p] = True
            continue
        while True:
            victim = buf[hand % cap]
            if ref.get(victim, False):
                ref[victim] = False
                hand += 1
            else:
                buf[hand % cap] = p
                ref[p] = True
                hand += 1
                break
    return misses


def bench_buffer_pool(seed: int = 20261231 + 434) -> dict[str, float]:
    rng = random.Random(seed)
    flush_ok = hot = 0
    ratio_sum = 0.0
    trials = 40
    for _ in range(trials):
        # skewed 80/20 trace
        hot_pages = list(range(3))
        trace = [
            rng.choice(hot_pages) if rng.random() < 0.8 else rng.randrange(3, 12)
            for _ in range(120)
        ]
        cap = 4
        m_fifo, m_clock = fifo_run(trace, cap), clock_run(trace, cap)
        ratio_sum += m_clock / max(1, m_fifo)
        # dirty-page flush semantics: every write must reach disk —
        # after the final flush, disk[p] equals p's total access count
        disk = {p: 0 for p in range(12)}
        buf: dict[int, int] = {}
        dirty: set[int] = set()
        hot_hit = hot_acc = 0
        for p in trace[:60]:
            resident = p in buf
            if not resident:
                if len(buf) >= cap:
                    victim = min(buf)
                    if victim in dirty:
                        disk[victim] = buf[victim]
                        dirty.discard(victim)
                    del buf[victim]
                buf[p] = disk[p]
            buf[p] += 1
            dirty.add(p)
            if p in hot_pages:
                hot_acc += 1
                hot_hit += int(resident)
        for p in dirty:
            disk[p] = buf[p]
        flush_ok += int(all(disk[p] == trace[:60].count(p) for p in range(12)))
        # real hit rate on hot-page accesses inside the pool sim
        hot += int(hot_acc > 0 and hot_hit / hot_acc >= 0.5)
    return {
        "synthetic_flush_preserves_writes": float(flush_ok / trials),
        "synthetic_hot_pages_hit": float(hot / trials),
        "synthetic_clock_miss_ratio": float(ratio_sum / trials),
    }
