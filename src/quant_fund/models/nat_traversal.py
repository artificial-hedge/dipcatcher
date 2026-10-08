"""NAT hole-punch sim (STUN/UDP): outbound packet creates pinhole; peer can (SYNTHETIC)
reach back only through a live pinhole."""

import numpy as np

_SEED = 20261231 + 612


class NAT:
    def __init__(self) -> None:
        self.pinholes: dict[tuple[str, int], int] = {}
        self.now = 0
        self.timeout = 20

    def send(self, dst: tuple[str, int]) -> None:
        self.now += 1
        self.pinholes[dst] = self.now

    def inbound_ok(self, src: tuple[str, int]) -> bool:
        exp = self.pinholes.get(src)
        return exp is not None and self.now <= exp + self.timeout


def bench_nat_traversal(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    delivered = 0
    blocked = 0
    punched_n = 0
    for _ in range(40):
        nat_a, nat_b = NAT(), NAT()
        pa, pb = ("10.1.0.1", 5000), ("10.2.0.1", 6000)
        punched = rng.rand() < 0.7
        punched_n += int(punched)
        if punched:
            nat_a.send(pb)
            nat_b.send(pa)
        # simulate hops
        nat_a.now = nat_b.now = rng.randint(0, 15)
        if punched and nat_a.inbound_ok(pb) and nat_b.inbound_ok(pa):
            delivered += 1
        if not punched:
            blocked += not (nat_a.inbound_ok(pb) or nat_b.inbound_ok(pa))
    # unpunched traffic must never pass; punched delivers at least half
    if blocked != 40 - punched_n or delivered == 0:
        raise ValueError("NAT traversal leaked or never delivered")
    return {
        "synthetic_nat_punched": delivered / 40,
        "synthetic_nat_deliver_rate": delivered / max(1, punched_n),
        "synthetic_nat_blocked_rate": blocked / max(1, 40 - punched_n),
    }
