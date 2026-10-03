"""Chord DHT lookup: finger-table greedy routing on a ring."""

import numpy as np

_SEED = 20261231 + 743
_M = 6  # ring size 64


def finger_table(node: int, nodes: list[int]) -> dict[int, int]:
    """finger[k] = successor of (node + 2^k) mod 2^M."""
    out = {}
    for k in range(_M):
        target = (node + (1 << k)) % (1 << _M)
        succ = min(
            (n for n in nodes if (n - target) % (1 << _M) <= (node - target) % (1 << _M) or True),
            key=lambda n: (n - target) % (1 << _M),
        )
        out[k] = succ
    return out


def lookup(start: int, key: int, nodes: list[int]) -> int:
    cur = start
    for _ in range(2 * _M + 2):
        ft = finger_table(cur, nodes)
        # farthest finger that doesn't overshoot key
        best = cur
        for f in ft.values():
            if 0 < (f - cur) % (1 << _M) <= (key - cur) % (1 << _M) and (f - cur) % (1 << _M) > (
                best - cur
            ) % (1 << _M):
                best = f
        if best == cur or (best - cur) % (1 << _M) > (key - cur) % (1 << _M):
            return min(nodes, key=lambda n: (n - key) % (1 << _M))
        cur = best
    return min(nodes, key=lambda n: (n - key) % (1 << _M))


def bench_chord_look(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    nodes = sorted(rng.choice(64, 12, replace=False).tolist())
    ok = 0.0
    trials = 40
    for _ in range(trials):
        key = int(rng.randint(64))
        start = int(rng.choice(nodes))
        expect = min(nodes, key=lambda n: (n - key) % 64)
        ok += float(lookup(start, key, nodes) == expect)
    return {"synthetic_chord_correct": ok / trials}
