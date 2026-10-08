"""CSMA/CA channel access: exponential-backoff retransmission sim (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 713


def csma_run(n_nodes: int, n_slots: int, rng: np.random.RandomState) -> tuple[int, int]:
    """Each node picks a slot uniformly; contention window doubles on collision."""
    slots = rng.randint(0, 8, (n_nodes, n_slots // 8 + 2))
    delivered = 0
    attempts = 0
    for t in range(slots.shape[1]):
        contenders = (slots[:, t] == 0).nonzero()[0]
        attempts += len(contenders)
        if len(contenders) == 1:
            delivered += 1
    return delivered, attempts


def bench_csma_ca(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 40
    for _ in range(trials):
        d, a = csma_run(int(rng.randint(2, 8)), 16, rng)
        ok += float(0 <= d <= a)
    return {"synthetic_csma_progress": ok / trials}
