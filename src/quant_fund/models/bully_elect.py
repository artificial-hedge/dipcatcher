"""Bully election: highest-id live node wins."""

import numpy as np

_SEED = 20261231 + 742


def bully(live: list[int], initiator: int) -> int:
    """Initiator sends election to higher ids; they answer; highest wins."""
    higher = [n for n in live if n > initiator]
    return max(live) if not higher else max(higher)


def bench_bully_elect(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 50
    for _ in range(trials):
        n = int(rng.randint(3, 12))
        alive = sorted(rng.choice(n, int(rng.randint(1, n)), replace=False).tolist())
        init = int(rng.choice(alive))
        ok += float(bully(alive, init) == max(alive))
    return {"synthetic_bully_max": ok / trials}
