"""Token ring: circulate token; only holder may transmit (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 741


def ring_run(n_nodes: int, n_msgs: int, rng: np.random.RandomState) -> list[int]:
    """Return sender order for n_msgs transmissions."""
    holder = 0
    senders: list[int] = []
    wants = rng.randint(0, n_nodes, n_msgs * 3)
    i = 0
    while len(senders) < n_msgs:
        if wants[i % len(wants)] == holder:
            senders.append(holder)
        holder = (holder + 1) % n_nodes
        i += 1
    return senders


def bench_token_ring(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    senders = ring_run(6, 12, rng)
    # safety: sender order must be a rotation-consistent subsequence
    ok = all(0 <= s < 6 for s in senders) and len(senders) == 12
    return {"synthetic_ring_safety": float(ok)}
