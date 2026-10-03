"""Longest/heaviest-chain fork resolution — SYNTHETIC.

Chains of (prev, work) blocks; verified: highest cumulative work wins,
common prefix is the fork point, tie → first seen.
"""

from __future__ import annotations


class Node:
    def __init__(self, i: int, prev: int | None, work: int) -> None:
        self.i, self.prev, self.work = i, prev, work


def best_tip(blocks: dict[int, Node]) -> int | None:
    """Cumulative work argmax; ties → lowest insert order wins (first seen)."""
    cum: dict[int, int] = {}

    def cw(i: int) -> int:
        if i in cum:
            return cum[i]
        n = blocks[i]
        cum[i] = n.work + (cw(n.prev) if n.prev is not None else 0)
        return cum[i]

    best: int | None = None
    for i in blocks:
        if best is None or cw(i) > cw(best):
            best = i
    return best


def chain(blocks: dict[int, Node], tip: int | None) -> list[int]:
    out = []
    while tip is not None:
        out.append(tip)
        tip = blocks[tip].prev
    return out[::-1]


def bench_fork_resolution(seed: int = 20261231 + 344) -> dict[str, float]:
    _ = seed
    heavier = prefix = tie = 0
    trials = 20
    for _ in range(trials):
        blocks: dict[int, Node] = {0: Node(0, None, 1)}
        # main chain 1..5 work 1 each
        for i in range(1, 6):
            blocks[i] = Node(i, i - 1, 1)
        # fork at 2 → heavier chain 100..104 work 2
        for i in range(100, 105):
            blocks[i] = Node(i, i - 1 if i > 100 else 2, 2)
        tip = best_tip(blocks)
        heavier += int(tip == 104)
        c = chain(blocks, tip)
        prefix += int(c[:3] == [0, 1, 2])
        # tie: two equal chains → first inserted wins
        b2: dict[int, Node] = {0: Node(0, None, 1), 1: Node(1, 0, 1), 2: Node(2, 0, 1)}
        tie += int(best_tip(b2) == 1)
    return {
        "synthetic_heaviest_wins": float(heavier / trials),
        "synthetic_common_prefix": float(prefix / trials),
        "synthetic_tie_first_seen": float(tie / trials),
    }
