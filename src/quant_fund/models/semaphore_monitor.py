"""SYNTHETIC counting semaphore + bounded-buffer producer/consumer.

Semaphore with wait/signal counters drives a capacity-N buffer;
verified: buffer bound never violated, no item lost/duplicated, all
produced items consumed (sequential interleaving model).
"""

from __future__ import annotations

import random


class Sem:
    def __init__(self, n: int):
        self.n = n

    def wait(self) -> bool:
        if self.n > 0:
            self.n -= 1
            return True
        return False

    def signal(self) -> None:
        self.n += 1


def run_bounded_buffer(produced: list[int], cap: int, rng: random.Random) -> tuple[list[int], bool]:
    buf: list[int] = []
    empty = Sem(cap)
    full = Sem(0)
    consumed: list[int] = []
    i = 0
    ok = True
    while i < len(produced) or buf:
        # interleave produce/consume randomly
        if i < len(produced) and (rng.random() < 0.5 or not buf):
            if empty.wait():
                buf.append(produced[i])
                full.signal()
                i += 1
                if len(buf) > cap:
                    ok = False
            else:
                if not full.wait():
                    raise RuntimeError("deadlock")
                consumed.append(buf.pop(0))
                empty.signal()
        else:
            if full.wait():
                consumed.append(buf.pop(0))
                empty.signal()
            else:
                if not empty.wait():
                    raise RuntimeError("deadlock")
                buf.append(produced[i])
                full.signal()
                i += 1
    return consumed, ok


def bench_semaphore_monitor(seed: int = 20261231 + 474) -> dict[str, float]:
    rng = random.Random(seed)
    bound = conserved = order = 0
    trials = 40
    for _ in range(trials):
        items = list(range(rng.randrange(5, 40)))
        consumed, ok = run_bounded_buffer(items, rng.randrange(1, 8), rng)
        bound += int(ok)
        conserved += int(sorted(consumed) == items)
        order += int(consumed == items)
    return {
        "synthetic_buffer_bound_held": float(bound / trials),
        "synthetic_no_loss_no_dup": float(conserved / trials),
        "synthetic_fifo_order": float(order / trials),
    }
