"""Traceroute path discovery: TTL-expiry hop enumeration (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 714


def traceroute(path: list[int], max_ttl: int = 30) -> list[int]:
    """Each hop replies with its id when TTL expires at it."""
    seen = []
    for ttl in range(1, max_ttl + 1):
        if ttl > len(path):
            break
        seen.append(path[ttl - 1])
        if ttl == len(path):
            break
    return seen


def bench_icmp_path(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 40
    for _ in range(trials):
        path = list(rng.randint(0, 1 << 16, int(rng.randint(2, 12))))
        got = traceroute(path)
        ok += float(got == path)
    return {"synthetic_traceroute_exact": ok / trials}
