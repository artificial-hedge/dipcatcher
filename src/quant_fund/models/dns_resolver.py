"""Recursive DNS resolver sim: cache + TTL expiry + delegation."""

import numpy as np

_SEED = 20261231 + 611


class Resolver:
    def __init__(self, zones: dict[str, str], ttl: int = 10) -> None:
        self.zones = zones
        self.ttl = ttl
        self.cache: dict[str, tuple[str, int]] = {}
        self.now = 0
        self.queries = 0

    def resolve(self, name: str) -> str | None:
        self.now += 1
        if name in self.cache:
            val, exp = self.cache[name]
            if self.now <= exp:
                return val
            del self.cache[name]
        self.queries += 1
        ans = self.zones.get(name)
        if ans is not None:
            self.cache[name] = (ans, self.now + self.ttl)
        return ans


def bench_dns_resolver(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    zones = {f"host{i}.example.com": f"10.0.0.{i}" for i in range(8)}
    r = Resolver(zones, ttl=10)
    correct = 0
    total = 0
    cached_correct = 0
    for _ in range(60):
        name = f"host{rng.randint(8)}.example.com"
        got = r.resolve(name)
        want = zones[name]
        total += 1
        correct += got == want
        if r.queries < total:
            cached_correct += 1
    return {
        "synthetic_dns_correct": correct / total,
        "synthetic_dns_cached": cached_correct / total,
    }
