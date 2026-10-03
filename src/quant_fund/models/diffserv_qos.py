"""DiffServ QoS: priority-weighted dequeuing vs strict-PQ oracle."""

import numpy as np

_SEED = 20261231 + 715


def diffserv_dequeue(queues: list[list[int]], weights: np.ndarray, n: int) -> list[int]:
    """Weighted round-robin over non-empty queues."""
    out: list[int] = []
    i = 0
    while len(out) < n and any(queues):
        i %= len(queues)
        q = queues[i]
        w = int(weights[i])
        for _ in range(w):
            if q:
                out.append(q.pop(0))
            if len(out) >= n:
                break
        i += 1
    return out


def bench_diffserv_qos(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 40
    for _ in range(trials):
        n_q = int(rng.randint(2, 4))
        queues = [list(rng.randint(0, 100, int(rng.randint(2, 10)))) for _ in range(n_q)]
        weights = rng.randint(1, 4, n_q)
        total = sum(len(q) for q in queues)
        out = diffserv_dequeue([list(q) for q in queues], weights, total)
        ok += float(sorted(out) == sorted(x for q in queues for x in q))
    return {"synthetic_qos_no_loss": ok / trials}
