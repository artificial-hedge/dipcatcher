"""Generational GC card table + write barrier (SYNTHETIC).

Old generation is split into fixed-size cards; a write barrier dirties
the card of the slot written. Minor GC scans only dirty cards to find
old->young references (remembered set).
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 896

_CARD = 16  # slots per card


class Heap:
    """Old+young heap of tagged slots: (gen, ref) where gen in {0=young,1=old}."""

    def __init__(self, n_old: int, n_young: int) -> None:
        self.old = np.full(n_old, -1, dtype=np.int64)  # refs into young or -1
        self.young = np.full(n_young, -1, dtype=np.int64)
        self.cards = np.zeros((n_old + _CARD - 1) // _CARD, dtype=bool)

    def write_old(self, slot: int, ref: int) -> None:
        """Write barrier: dirty the card covering `slot`."""
        self.old[slot] = ref
        self.cards[slot // _CARD] = True

    def remember(self) -> list[int]:
        """Scan dirty cards only; return young indices referenced from old."""
        found: list[int] = []
        for c in np.flatnonzero(self.cards):
            ci = int(c)
            lo, hi = ci * _CARD, min((ci + 1) * _CARD, self.old.size)
            for slot in range(lo, hi):
                r = int(self.old[slot])
                if r >= 0:
                    found.append(r)
            self.cards[c] = False
        return sorted(set(found))


def bench_card_table_gc(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n_old, n_young = 4096, 256
    h = Heap(n_old, n_young)
    # 300 writes into random old slots, some young refs
    slots = rng.choice(n_old, 300, replace=False)
    truth: set[int] = set()
    for s in slots:
        if rng.random() < 0.6:
            r = int(rng.integers(0, n_young))
            h.write_old(int(s), r)
            truth.add(r)
        else:
            h.write_old(int(s), -1)
    rem = set(h.remember())
    score = 0.0
    score += 1.0 if rem == truth else 0.0
    # no misses after barrier-elided writes too (conservative correctness):
    # write without barrier must NOT be found (proves scan is card-driven)
    h.young[:] = -1
    s2 = int(rng.integers(0, n_old))
    h.old[s2] = 7  # raw write, no barrier
    score += 1.0 if h.remember() == [] else 0.0
    # scan cost: only dirty cards touched (count dirty < total)
    score += 1.0 if int(np.flatnonzero(h.cards).size) == 0 else 0.0
    # second round: two writes same card -> one scan
    h.write_old(10, 3)
    h.write_old(11, 4)
    rem2 = h.remember()
    score += 1.0 if rem2 == [3, 4] else 0.0
    return {"synthetic_card_table_gc": score / 4.0}
