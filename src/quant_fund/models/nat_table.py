"""SYNTHETIC NAT translation table.

Inside (ip,port) → outside port mapping; reverse table consistency,
distinct outsides for distinct insides (no collisions), exhaustion when
port space is full.
"""

from __future__ import annotations

import random


class NAT:
    def __init__(self, lo: int = 40000, hi: int = 40010):
        self.lo, self.hi = lo, hi
        self.fwd: dict[tuple[str, int], int] = {}
        self.rev: dict[int, tuple[str, int]] = {}

    def translate(self, ip: str, port: int) -> int | None:
        key = (ip, port)
        if key in self.fwd:
            return self.fwd[key]
        used = set(self.rev)
        for p in range(self.lo, self.hi + 1):
            if p not in used:
                self.fwd[key] = p
                self.rev[p] = key
                return p
        return None  # exhausted

    def inbound(self, out_port: int) -> tuple[str, int] | None:
        return self.rev.get(out_port)


def bench_nat_table(seed: int = 20261231 + 404) -> dict[str, float]:
    rng = random.Random(seed)
    stable = unique = exhaust = reverse = 0
    trials = 40
    for _ in range(trials):
        nat = NAT(5000, 5000 + rng.randrange(5, 15))
        cap = nat.hi - nat.lo + 1
        n = cap + rng.randrange(0, 5)
        ins = [(f"10.0.0.{rng.randrange(1, 255)}", rng.randrange(1024, 60000)) for _ in range(n)]
        ins = list(dict.fromkeys(ins))
        outs = [nat.translate(*k) for k in ins]
        got = [o for o in outs if o is not None]
        # stable: same inside → same outside
        stable += int(
            all(nat.translate(*k) == o for k, o in zip(ins, outs, strict=True) if o is not None)
        )
        # unique: no two insides share an outside
        unique += int(len(set(got)) == len(got))
        # exhaustion iff more distinct insides than ports
        exhaust += int(len(got) == min(len(ins), cap))
        # reverse: inbound(out) returns the original inside
        reverse += int(
            all(nat.inbound(o) == k for k, o in zip(ins, outs, strict=True) if o is not None)
        )
    return {
        "synthetic_stable_mapping": float(stable / trials),
        "synthetic_no_collisions": float(unique / trials),
        "synthetic_exhaustion_exact": float(exhaust / trials),
        "synthetic_reverse_lookup": float(reverse / trials),
    }
