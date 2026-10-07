"""MESI cache-coherence protocol simulator — state-transition invariant check (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 641

_STATES = ("M", "E", "S", "I")


class MesiSim:
    """Two-core MESI: 'R' reads, 'W' writes a shared line."""

    def __init__(self) -> None:
        self.s0 = "I"
        self.s1 = "I"
        self.mem = 0
        self.c0 = 0
        self.c1 = 0

    def read0(self) -> None:
        if self.s0 in ("M", "E", "S"):
            return
        if self.s1 == "M":
            self.mem = self.c1
            self.s1 = "S"
            self.c0 = self.c1
            self.s0 = "S"
        elif self.s1 in ("E", "S"):
            self.c0 = self.c1
            self.s0 = "S"
            self.s1 = "S"
        else:
            self.c0 = self.mem
            self.s0 = "E"

    def write0(self, v: int) -> None:
        if self.s0 == "M":
            self.c0 = v
            return
        # need exclusive ownership
        if self.s1 == "M":
            self.mem = self.c1
        self.s1 = "I"
        self.c0 = v
        self.s0 = "M"

    def read1(self) -> None:
        if self.s1 in ("M", "E", "S"):
            return
        if self.s0 == "M":
            self.mem = self.c0
            self.s0 = "S"
            self.c1 = self.c0
            self.s1 = "S"
        elif self.s0 in ("E", "S"):
            self.c1 = self.c0
            self.s1 = "S"
            self.s0 = "S"
        else:
            self.c1 = self.mem
            self.s1 = "E"

    def write1(self, v: int) -> None:
        if self.s1 == "M":
            self.c1 = v
            return
        if self.s0 == "M":
            self.mem = self.c0
        self.s0 = "I"
        self.c1 = v
        self.s1 = "M"


def _invariant(sim: MesiSim) -> bool:
    # coherence invariant: at most one writer (M); S lines all share same value;
    # if both read, values agree with mem as far as protocol allows
    if sim.s0 == "M" and sim.s1 == "M":
        return False
    return not (sim.s0 == "S" and sim.s1 == "S" and sim.c0 != sim.c1)


def bench_mesi_cache(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    trials = 60
    for _ in range(trials):
        sim = MesiSim()
        good = True
        for _ in range(40):
            op = rng.randint(4)
            v = rng.randint(100)
            if op == 0:
                sim.read0()
            elif op == 1:
                sim.write0(v)
            elif op == 2:
                sim.read1()
            else:
                sim.write1(v)
            if not _invariant(sim):
                good = False
                break
        ok += good
    return {"synthetic_mesi_invariant": ok / trials}
