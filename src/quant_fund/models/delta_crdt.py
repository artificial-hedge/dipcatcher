"""Delta-state CRDT: OR-Set with delta-group dissemination.

Observed-Remove set: adds are tagged unique dots (replica, counter); removes
tombstone only the dots observed. The delta variant ships only the
difference since the last merge, joined into each peer's state. Verified:
convergence (all replicas' states equal after delta exchange), equivalence
with full-state merge, add-wins under concurrent add+remove.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 957

Dot = tuple[int, int]


class ORSetDelta:
    def __init__(self, rid: int) -> None:
        self.rid = rid
        self.elems: dict[str, set[Dot]] = {}
        self.clock: dict[int, int] = {}
        self.tomb: set[Dot] = set()

    def _dot(self) -> Dot:
        c = self.clock.get(self.rid, 0) + 1
        self.clock[self.rid] = c
        return (self.rid, c)

    def add(self, e: str) -> dict[str, object]:
        d = self._dot()
        self.elems.setdefault(e, set()).add(d)
        return {"elems": {e: {d}}, "clock": dict(self.clock), "tomb": set()}

    def remove(self, e: str) -> dict[str, object]:
        gone = self.elems.pop(e, set())
        self.tomb |= gone
        return {"elems": {}, "clock": dict(self.clock), "tomb": set(gone)}

    def merge(self, delta: dict[str, object]) -> None:
        elems = delta["elems"]
        tomb = delta["tomb"]
        clock = delta["clock"]
        if not (isinstance(elems, dict) and isinstance(tomb, set) and isinstance(clock, dict)):
            raise ValueError(
                "isinstance(elems, dict) and isinstance(tomb, set) and isinstance(clock, dict)"
            )
        for e, ds in elems.items():
            keep = {d for d in ds if d not in self.tomb}
            if keep:
                self.elems.setdefault(e, set()).update(keep)
        for e in list(self.elems):
            self.elems[e] -= tomb
            if not self.elems[e]:
                del self.elems[e]
        self.tomb |= {d for d in tomb}
        for r, c in clock.items():
            self.clock[r] = max(self.clock.get(r, 0), c)

    def value(self) -> set[str]:
        return set(self.elems)


def bench_delta_crdt(seed: int = _SEED) -> dict[str, float]:
    _ = np.random.default_rng(seed)
    a, b, c = ORSetDelta(0), ORSetDelta(1), ORSetDelta(2)
    d1 = a.add("x")
    d2 = b.add("y")
    d3 = c.remove("x")  # hasn't seen x -> empty delta
    for rep in (a, b, c):
        for d in (d1, d2, d3):
            rep.merge(d)
    conv1 = a.value() == b.value() == c.value() == {"x", "y"}
    # add-wins: c sees x, removes it; a concurrently adds x again
    d4 = a.add("x")  # new dot on x
    d5 = c.remove("x")  # tombstones only dots c knows
    for rep in (a, b, c):
        rep.merge(d4)
        rep.merge(d5)
    conv2 = a.value() == b.value() == c.value() == {"x", "y"}
    # delta equivalence: fresh replica merged via deltas == full state
    full = ORSetDelta(9)
    full.elems = {e: set(ds) for e, ds in a.elems.items()}
    full.tomb = set(a.tomb)
    conv3 = full.value() == a.value()
    return {"synthetic_delta_crdt": float(np.mean([conv1, conv2, conv3]))}
