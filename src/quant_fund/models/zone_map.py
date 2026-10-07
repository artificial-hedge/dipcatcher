"""Zone-map skip-scan pruning: per-block (min,max) enables block skipping (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 602


def _zone_scan(data: np.ndarray, lo: float, hi: float, block: int) -> tuple[np.ndarray, int]:
    out: list[float] = []
    scanned = 0
    for s in range(0, len(data), block):
        blk = data[s : s + block]
        if blk.max() < lo or blk.min() > hi:
            continue  # pruned
        scanned += 1
        out.extend(blk[(blk >= lo) & (blk <= hi)])
    return np.asarray(out), scanned


def bench_zone_map(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    eq = 0
    pruned_any = 0
    for _ in range(40):
        n = 600
        data = np.sort(rng.uniform(0, 1, n)) if rng.rand() < 0.7 else rng.uniform(0, 1, n)
        lo, hi = sorted(rng.uniform(0.1, 0.9, 2))
        got, scanned = _zone_scan(data, lo, hi, 30)
        want = data[(data >= lo) & (data <= hi)]
        if np.allclose(np.sort(got), np.sort(want)):
            eq += 1
        if scanned < n / 30:
            pruned_any += 1
    return {
        "synthetic_zone_equiv": eq / 40,
        "synthetic_zone_pruned": pruned_any / 40,
    }
