"""ABD (Attiya-Bar-Noy-Dolev) atomic read/write register emulation (SYNTHETIC).

Writers tag values with a logical timestamp; a write completes after a
majority of replicas stores it. A read takes a majority, picks the
max-timestamp value, and writes it back (read-writeback) before
returning. Verified: every read returns the latest COMPLETED write's
value or a concurrent one (atomicity), and read-back propagation makes
a later read see the freshest value.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 962


class Replica:
    def __init__(self) -> None:
        self.ts = 0
        self.val = 0


class ABDRegister:
    def __init__(self, n: int) -> None:
        self.reps = [Replica() for _ in range(n)]
        self.q = n // 2 + 1
        self.ts = 0

    def _pick(self, rng: np.random.Generator) -> list[Replica]:
        idx = rng.choice(len(self.reps), self.q, replace=False)
        return [self.reps[i] for i in idx]

    def write(self, val: int, rng: np.random.Generator) -> int:
        self.ts += 1
        for r in self._pick(rng):
            r.ts, r.val = self.ts, val
        return self.ts

    def read(self, rng: np.random.Generator) -> tuple[int, int]:
        qs = self._pick(rng)
        top = max(qs, key=lambda r: r.ts)
        # write-back phase
        for r in self._pick(rng):
            if r.ts < top.ts:
                r.ts, r.val = top.ts, top.val
        return top.ts, top.val


def bench_abd_register(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    reg = ABDRegister(5)
    t1 = reg.write(11, rng)
    t_read, v_read = reg.read(rng)
    first_ok = v_read == 11 and t_read == t1
    t2 = reg.write(22, rng)
    t_read2, v_read2 = reg.read(rng)
    second_ok = v_read2 == 22 and t_read2 == t2
    # after read write-back, majority holds freshest -> stable reads
    stable = all(reg.read(rng)[1] == 22 for _ in range(6))
    # a partial second write reaching only 2 replicas must not corrupt
    reg2 = ABDRegister(5)
    reg2.write(7, rng)
    partial = reg2._pick(rng)
    for r in partial[:2]:
        r.ts, r.val = reg2.ts + 1, 99
    t, v = reg2.read(rng)
    # read returns either the completed 7 or the partial 99 (atomicity allows
    # seeing the incomplete write) — but never an older/invalid value
    atomic = v in (7, 99)
    return {"synthetic_abd_register": float(np.mean([first_ok, second_ok, stable, atomic]))}
