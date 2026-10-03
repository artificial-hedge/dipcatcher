"""ARP cache sim: request/reply learning, timeout eviction, gratuitous update."""

import numpy as np

_SEED = 20261231 + 613


class ArpCache:
    def __init__(self, ttl: int = 15) -> None:
        self.table: dict[str, tuple[str, int]] = {}
        self.ttl = ttl
        self.now = 0

    def learn(self, ip: str, mac: str) -> None:
        self.table[ip] = (mac, self.now)

    def lookup(self, ip: str) -> str | None:
        e = self.table.get(ip)
        if e is None:
            return None
        if self.now - e[1] > self.ttl:
            del self.table[ip]
            return None
        return e[0]


def bench_arp_table(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    truth = {f"10.0.0.{i}": f"aa:bb:{i:02x}" for i in range(8)}
    arp = ArpCache(ttl=15)
    ok = 0
    total = 0
    for _ in range(80):
        arp.now += rng.randint(0, 6)
        ip = f"10.0.0.{rng.randint(8)}"
        if rng.rand() < 0.5:
            arp.learn(ip, truth[ip])
        got = arp.lookup(ip)
        fresh = ip in arp.table and arp.now - arp.table[ip][1] <= arp.ttl
        total += 1
        ok += (got == truth[ip]) if fresh else (got is None)
    return {"synthetic_arp_correct": ok / total}
