"""Virtual method dispatch: per-class vtables, single-inheritance override."""

import numpy as np

_SEED = 20261231 + 542


class Klass:
    def __init__(self, name: str, parent: "Klass | None") -> None:
        self.name = name
        self.parent = parent
        self.methods: dict[str, str] = {}
        self.vtable: dict[str, str] = dict(parent.vtable) if parent else {}

    def define(self, slot: str, impl: str) -> None:
        self.methods[slot] = impl
        self.vtable[slot] = impl


class Instance:
    def __init__(self, klass: Klass) -> None:
        self.klass = klass

    def call(self, slot: str) -> str:
        return f"{self.klass.vtable[slot]}@{self.klass.name}"


def bench_dispatch_table(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    A = Klass("A", None)
    A.define("m0", "a0")
    A.define("m1", "a1")
    B = Klass("B", A)
    B.define("m1", "b1")
    C = Klass("C", B)
    C.define("m2", "c2")
    insts = [Instance(A), Instance(B), Instance(C)] * 20
    order = list(range(len(insts)))
    rng.shuffle(order)
    insts = [insts[i] for i in order]
    hits = 0
    expect = {
        "A": {"m0": "a0@A", "m1": "a1@A"},
        "B": {"m0": "a0@B", "m1": "b1@B"},
        "C": {"m0": "a0@C", "m1": "b1@C", "m2": "c2@C"},
    }
    for inst in insts:
        for slot in ("m0", "m1", "m2"):
            if slot in inst.klass.vtable:
                hits += int(inst.call(slot) == expect[inst.klass.name][slot])
    total = sum(1 for inst in insts for s in ("m0", "m1", "m2") if s in inst.klass.vtable)

    # oracle: walk chain manually
    def oracle(inst: Instance, slot: str) -> str:
        k: Klass | None = inst.klass
        while k is not None:
            if slot in k.methods:
                return f"{k.methods[slot]}@{inst.klass.name}"
            k = k.parent
        raise KeyError(slot)

    oracle_hits = sum(
        1
        for inst in insts
        for s in ("m0", "m1", "m2")
        if s in inst.klass.vtable and inst.call(s) == oracle(inst, s)
    )
    return {
        "synthetic_dispatch_exact": float(hits / total),
        "synthetic_matches_oracle": float(oracle_hits / total),
        "synthetic_override_correct": float(Instance(C).call("m1") == "b1@C"),
    }
