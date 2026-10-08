"""CSMA/CA channel access: exponential-backoff retransmission sim (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 713


def csma_run(n_nodes: int, n_slots: int, rng: np.random.RandomState) -> tuple[int, int, int]:
    """CSMA/CA exponential backoff: each node holds one packet, decrements
    its backoff counter every slot, and transmits at 0. A sole transmission
    is delivered; a collision doubles each collider's contention window
    (capped at 1024) and it redraws. Returns (delivered, attempts, collisions).
    """
    cw = np.full(n_nodes, 8)
    backoff = rng.randint(0, 8, n_nodes)
    pending = np.ones(n_nodes, dtype=bool)
    delivered = attempts = collisions = 0
    for _ in range(n_slots):
        contenders = np.nonzero(pending & (backoff == 0))[0]
        attempts += len(contenders)
        if len(contenders) == 1:
            delivered += 1
            pending[contenders[0]] = False
        elif len(contenders) > 1:
            collisions += 1
        for j in contenders:
            if len(contenders) > 1:
                cw[j] = min(cw[j] * 2, 1024)
            backoff[j] = int(rng.randint(0, cw[j]))
        # everyone else's countdown ticks down this slot
        idle = pending & (backoff > 0)
        idle[contenders] = False
        backoff[idle] -= 1
    return delivered, attempts, collisions


def bench_csma_ca(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    checks = []
    frac = att = 0.0
    trials = 40
    for _ in range(trials):
        n_nodes = int(rng.randint(2, 8))
        d, a, _c = csma_run(n_nodes, 64, rng)
        checks.append(0 <= d <= a)  # a delivery needs an attempt
        checks.append(d <= n_nodes)  # each node carries one packet
        checks.append(d >= 1)  # backoff makes progress
        frac += d / n_nodes
        att += a
    return {
        "synthetic_csma_progress": sum(checks) / (3 * trials),
        "synthetic_csma_delivered_frac": frac / trials,
        "synthetic_csma_attempts_mean": att / trials,
    }
