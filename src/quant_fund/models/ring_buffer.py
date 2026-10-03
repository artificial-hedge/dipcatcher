"""SPSC lock-free ring buffer: no loss/reorder under interleaved ops."""

import numpy as np

_SEED = 20261231 + 655


class SpscRing:
    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.buf = [0] * cap
        self.head = 0
        self.tail = 0

    def push(self, v: int) -> bool:
        nxt = (self.tail + 1) % self.cap
        if nxt == self.head:
            return False
        self.buf[self.tail] = v
        self.tail = nxt
        return True

    def pop(self) -> int | None:
        if self.head == self.tail:
            return None
        v = self.buf[self.head]
        self.head = (self.head + 1) % self.cap
        return v


def bench_ring_buffer(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    trials = 50
    for _ in range(trials):
        ring = SpscRing(rng.randint(4, 10))
        produced = list(range(30))
        consumed: list[int] = []
        i = 0
        while i < len(produced) or consumed != produced[: len(consumed)]:
            if i < len(produced) and rng.rand() < 0.6:
                ring.push(produced[i])
                i += 1
            v = ring.pop()
            if v is not None:
                consumed.append(v)
            if len(consumed) == len(produced):
                break
        ok += consumed == produced
    return {"synthetic_ring_order": ok / trials}
