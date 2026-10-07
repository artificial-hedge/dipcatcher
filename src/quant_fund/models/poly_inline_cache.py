"""Polymorphic inline cache: monomorphic → polymorphic → megamorphic states (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 543


class PIC:
    MAX_POLY = 4

    def __init__(self) -> None:
        self.entries: dict[str, str] = {}
        self.state = "mono"
        self.misses = 0

    def dispatch(self, klass: str, method: str, vtables: dict[str, dict[str, str]]) -> str:
        if self.state == "mega":
            return vtables[klass][method]
        if klass in self.entries:
            return self.entries[klass]
        self.misses += 1
        impl = vtables[klass][method]
        self.entries[klass] = impl
        if len(self.entries) == 1:
            self.state = "mono"
        elif len(self.entries) <= self.MAX_POLY:
            self.state = "poly"
        else:
            self.state = "mega"
            self.entries.clear()
        return impl


def bench_poly_inline_cache(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    klasses = {f"K{i}": {"m": f"impl{i}"} for i in range(8)}
    pic = PIC()
    # workload: mostly K0/K1, rare others
    seq = [rng.choice([0, 0, 0, 1, 1, 2, 3]) for _ in range(400)]
    seq += [4, 5, 6, 7] + list(seq[:50])
    correct = 0
    for k in seq:
        got = pic.dispatch(f"K{k}", "m", klasses)
        correct += int(got == f"impl{k}")
    # second PIC on a diverse stream should go megamorphic
    pic2 = PIC()
    for k in range(8):
        pic2.dispatch(f"K{k}", "m", klasses)
    mega = pic2.state == "mega"
    # megamorphic still correct
    pic3 = PIC()
    seq3 = [i % 8 for i in range(200)]
    pic3_hits = sum(pic3.dispatch(f"K{k}", "m", klasses) == f"impl{k}" for k in seq3)
    return {
        "synthetic_dispatch_correct": float(correct / len(seq)),
        "synthetic_mega_transition": float(mega),
        "synthetic_mega_correct": float(pic3_hits / len(seq3)),
    }
