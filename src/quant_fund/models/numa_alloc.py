"""NUMA first-touch allocation: local vs remote access ratio metric."""

import numpy as np

_SEED = 20261231 + 645


def bench_numa_alloc(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    local_ratios = []
    for _ in range(30):
        n_nodes = 4
        # each node first-touches its own region
        region = {i: rng.randint(0, 1000, 50) + i * 10000 for i in range(n_nodes)}
        accesses = rng.randint(0, n_nodes, 200)
        local = 0
        for a in accesses:
            # node a primarily accesses its own region (80% affinity)
            tgt = region[a] if rng.rand() < 0.8 else region[rng.randint(n_nodes)]
            local += any(abs(t - a * 10000) < 10000 for t in tgt)
        local_ratios.append(local / 200)
    return {"synthetic_numa_locality": float(np.mean(local_ratios))}
