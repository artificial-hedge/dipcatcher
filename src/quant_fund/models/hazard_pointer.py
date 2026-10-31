"""Hazard-pointer reclamation simulator (Michael 2004): retired objects are (SYNTHETIC)
freed only when no thread's hazard slot protects them."""

import numpy as np

_SEED = 20261231 + 580


class HazardSim:
    def __init__(self, n_threads: int = 4) -> None:
        self.hazard: dict[int, int | None] = {t: None for t in range(n_threads)}
        self.retired: set[int] = set()
        self.freed: set[int] = set()

    def protect(self, tid: int, obj: int) -> None:
        self.hazard[tid] = obj

    def clear(self, tid: int) -> None:
        self.hazard[tid] = None

    def retire(self, obj: int) -> None:
        if obj not in self.freed:
            self.retired.add(obj)

    def collect(self) -> set[int]:
        protected = {h for h in self.hazard.values() if h is not None}
        to_free = self.retired - protected
        self.freed |= to_free
        self.retired -= to_free
        return to_free


def _oracle_trace(rng: np.random.RandomState) -> bool:
    sim = HazardSim()
    n_obj = 0
    for _ in range(300):
        r = rng.rand()
        tid = rng.randint(4)
        if r < 0.35:
            sim.protect(tid, rng.randint(max(n_obj, 1)))
        elif r < 0.55:
            sim.clear(tid)
        elif r < 0.8:
            sim.retire(n_obj)
            n_obj += 1
        else:
            protected = {h for h in sim.hazard.values() if h is not None}
            expected = sim.retired - protected
            actual = sim.collect()
            if actual != expected or actual & protected:
                return False
    return True


def bench_hazard_pointer(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = sum(_oracle_trace(np.random.RandomState(rng.randint(2**31))) for _ in range(30))
    return {"synthetic_hazard_safety": ok / 30}
