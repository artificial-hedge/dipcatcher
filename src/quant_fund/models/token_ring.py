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
    n_nodes = 6
    n_msgs = 12
    senders = ring_run(n_nodes, n_msgs, rng)
    # Reference oracle: replay the same seeded wants stream through an
    # independent simulation — catches out-of-turn sends that a pure
    # range/consistency check cannot (any sender sequence is cyclically
    # consistent).
    rng2 = np.random.RandomState(seed)
    wants = rng2.randint(0, n_nodes, n_msgs * 3)
    ref: list[int] = []
    holder, i = 0, 0
    while len(ref) < n_msgs:
        if wants[i % len(wants)] == holder:
            ref.append(holder)
        holder = (holder + 1) % n_nodes
        i += 1
    ok = senders == ref and all(0 <= s < n_nodes for s in senders)
    return {"synthetic_ring_safety": float(ok)}
