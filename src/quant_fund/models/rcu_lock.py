"""RCU (read-copy-update) grace-period simulator (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 585


class RCU:
    def __init__(self) -> None:
        self.version = 0
        self.readers: set[int] = set()  # reader snapshot versions
        self.retired: list[int] = []
        self.freed: list[int] = []

    def read_lock(self, rid: int) -> None:
        self.readers.add(rid)

    def read_unlock(self, rid: int) -> None:
        self.readers.discard(rid)

    def update(self) -> None:
        self.retired.append(self.version)
        self.version += 1

    def synchronize(self) -> int:
        """Grace period: wait until no pre-update readers remain.
        In the sim, returns count of reclaimable versions (all readers have
        already observed post-update state)."""
        if self.readers:
            return 0
        n = len(self.retired)
        self.freed.extend(self.retired)
        self.retired.clear()
        return n


def bench_rcu_lock(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    for _ in range(30):
        rcu = RCU()
        rid = 0
        safe = True
        for _ in range(200):
            r = rng.rand()
            if r < 0.4:
                rid += 1
                rcu.read_lock(rid)
            elif r < 0.6:
                if rcu.readers:
                    rcu.read_unlock(rng.choice(list(rcu.readers)))
            elif r < 0.9:
                rcu.update()
            else:
                freed = rcu.synchronize()
                if freed > 0 and rcu.readers:
                    safe = False
        if safe:
            ok += 1
    return {"synthetic_rcu_grace": ok / 30}
