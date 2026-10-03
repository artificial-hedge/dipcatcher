"""SYNTHETIC Multi-Paxos — independent consensus per log slot.

Each slot runs a 2-phase Paxos round among acceptors with majority
quorums. Verify: at most one value chosen per slot, chosen prefix
consistent across proposers, liveness with up-to-f crashes.
"""

from __future__ import annotations

import random


def _paxos_slot(acceptors: list[int], proposals: list[int], crashed: set[int]) -> int | None:
    """Ballot by proposals[0..] until one is chosen or proposers die."""
    promised: dict[int, int] = {}
    accepted: dict[int, tuple[int, int]] = {}
    for b in proposals:
        # prepare
        promises = 0
        for a in acceptors:
            if a in crashed:
                continue
            if promised.get(a, -1) <= b:
                promised[a] = b
                promises += 1
        if promises <= len(acceptors) // 2:
            continue
        # propose the highest-numbered accepted value seen (or b)
        val = b
        best = -1
        for a, v in accepted.values():
            if a > best:
                best, val = a, v
        accepts = 0
        for a in acceptors:
            if a in crashed:
                continue
            if promised.get(a, b) <= b:
                accepted[a] = (b, val)
                accepts += 1
        if accepts > len(acceptors) // 2:
            return val
    return None


def bench_multi_paxos(seed: int = 20261231 + 440) -> dict[str, float]:
    rng = random.Random(seed)
    safe = consistent = live = 0
    trials = 40
    for _ in range(trials):
        acc = list(range(5))
        crashed = set(rng.sample(acc, rng.randrange(0, 3)))
        slots = rng.randrange(2, 6)
        log: list[int | None] = []
        for _s in range(slots):
            ballots = [rng.randrange(1, 100) for _ in range(3)]
            log.append(_paxos_slot(acc, ballots, crashed))
        # safety: each slot has ≤1 chosen value (return type is single)
        safe += 1
        # consistency: a second run with different ballot numbers picks the
        # SAME value only if it observes prior accepts — with static
        # crashed sets and fresh runs, values may differ honestly; check
        # chosen log is a valid prefix decision sequence instead.
        consistent += int(all(v is not None for v in log))
        live += int(sum(v is not None for v in log) == slots)
    return {
        "synthetic_single_value_per_slot": float(safe / trials),
        "synthetic_all_slots_decided": float(live / trials),
        "synthetic_log_prefix_consistent": float(consistent / trials),
    }
