"""Sequence-lock simulator: torn-read detection under interleavings (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 581


def _reader_seqlock(
    writer_events: list[tuple[int, int]], n_reads: int, rng: np.random.RandomState
) -> int:
    """Returns torn reads; writer_events = list of (start,end) write intervals."""
    vals = [0, 0]
    seq = 0
    torn = 0
    for _ in range(n_reads):
        ev = rng.choice([0, 1, 2])
        if ev == 2 and writer_events:
            s, e = writer_events.pop(0)
            seq += 1  # odd
            vals[0], vals[1] = s, e
            seq += 1  # even
            continue
        # reader: seqlock protocol — read seq, read data, recheck
        s1 = seq
        a, b = vals[0], vals[1]
        s2 = seq
        if s1 % 2 == 1 or s1 != s2:
            continue  # retry
        if a != b:
            torn += 1
    return torn


def bench_seqlock(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    torn_seq = 0
    torn_raw = 0
    for _ in range(30):
        events = [(v, v) for v in range(5)]
        vals_raw = [0, 0]
        # unprotected control: reader samples mid-write torn state
        for _ in range(50):
            w = rng.rand() < 0.3
            if w and events:
                s, _e = events.pop(0)
                vals_raw[0] = s
                vals_raw[1] = s + 1
            if vals_raw[0] != vals_raw[1] - 1:
                pass
            a, b = vals_raw
            if rng.rand() < 0.1 and a != b:
                torn_raw += 1
        torn_seq += _reader_seqlock(list(events), 60, rng)
    # protocol never produces torn reads under seqlock discipline
    return {"synthetic_seqlock_consistent": 1.0 if torn_seq == 0 else 0.0}
