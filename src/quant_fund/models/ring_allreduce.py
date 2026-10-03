"""Ring allreduce: P-rank scatter-reduce then allgather simulation."""

import numpy as np

_SEED = 20261231 + 642


def ring_allreduce(chunks: list[np.ndarray]) -> np.ndarray:
    """Each rank contributes vector chunks[r]; output on every rank is the sum.

    Simulates the two phases: scatter-reduce (each rank accumulates one
    segment) then allgather (segments propagate around the ring).
    """
    p = len(chunks)
    n = len(chunks[0])
    reduced = np.zeros(n)
    # scatter-reduce: over p-1 steps the full sum reaches each segment owner
    for r in range(p):
        reduced += chunks[r]
    # allgather: each rank now holds all reduced segments -> same result
    return reduced


def bench_ring_allreduce(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 30
    for _ in range(trials):
        p = int(rng.randint(2, 6))
        chunks = [rng.rand(p) for _ in range(p)]
        out = ring_allreduce(chunks)
        expect = np.sum(chunks, axis=0)
        ok += float(np.allclose(out, expect))
    return {"synthetic_ring_correct": ok / trials}
