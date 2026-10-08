"""Epoch-based reclamation (EBR) simulator (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 583


class EBR:
    def __init__(self, n_threads: int = 4) -> None:
        self.epoch = 0
        self.local: dict[int, int | None] = {t: None for t in range(n_threads)}
        self.bags: list[set[int]] = [set(), set(), set()]
        self.freed: set[int] = set()

    def enter(self, tid: int) -> None:
        self.local[tid] = self.epoch

    def exit(self, tid: int) -> None:
        self.local[tid] = None

    def retire(self, obj: int) -> None:
        self.bags[self.epoch % 3].add(obj)

    def try_advance(self) -> int:
        if all(e is None or e >= self.epoch for e in self.local.values()):
            self.epoch += 1
            old = self.bags[self.epoch % 3]
            n = len(old)
            self.freed |= old
            self.bags[self.epoch % 3] = set()
            return n
        return 0


def _ebr_trace(rng: np.random.RandomState) -> bool:
    """Invariant: no object freed while a thread in an earlier epoch could
    still hold a reference. Modelled: freed ⊂ retired ≥2 epochs ago."""
    ebr = EBR()
    retired_at: dict[int, int] = {}
    oid = 0
    for _ in range(300):
        r = rng.rand()
        tid = rng.randint(4)
        if r < 0.3:
            ebr.enter(tid)
        elif r < 0.5:
            ebr.exit(tid)
        elif r < 0.8:
            retired_at[oid] = ebr.epoch
            ebr.retire(oid)
            oid += 1
        else:
            before = ebr.freed.copy()
            ebr.try_advance()
            for o in ebr.freed - before:
                # freed objects must have been retired >=2 epochs ago
                if ebr.epoch - retired_at[o] < 2:
                    return False
                # and no live critical section may predate its retirement
                if any(le is not None and le <= retired_at[o] for le in ebr.local.values()):
                    return False
    return True


def bench_epoch_reclaim(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = sum(_ebr_trace(np.random.RandomState(rng.randint(2**31))) for _ in range(25))
    return {"synthetic_ebr_safe": ok / 25}
