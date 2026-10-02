"""Learning Ethernet switch: MAC table, flood-on-unknown, aging."""

import numpy as np

_SEED = 20261231 + 615


class Switch:
    def __init__(self, ttl: int = 20) -> None:
        self.table: dict[str, tuple[int, int]] = {}
        self.ttl = ttl
        self.now = 0
        self.floods = 0
        self.unicasts = 0

    def frame(self, src: str, dst: str, port: int) -> bool:
        self.now += 1
        self.table[src] = (port, self.now)
        e = self.table.get(dst)
        if e is not None and self.now - e[1] <= self.ttl:
            self.unicasts += 1
            return True
        self.floods += 1
        return False


def bench_eth_switch(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    sw = Switch(ttl=100)
    hosts = {"a": 1, "b": 2, "c": 3, "d": 4}
    names = list(hosts)
    unicast_ratio_hits = 0
    n = 0
    for _ in range(60):
        src, dst = rng.choice(names, 2, replace=False)
        known = dst in sw.table and sw.now - sw.table[dst][1] <= sw.ttl
        direct = sw.frame(src, dst, hosts[src])
        n += 1
        unicast_ratio_hits += direct == known
    return {"synthetic_switch_learn": unicast_ratio_hits / n}
