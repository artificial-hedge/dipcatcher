"""GC skew: (G-C)/(G+C) sliding window; origin detection."""

import numpy as np

_SEED = 20261231 + 730


def gc_skew(seq: str, window: int) -> np.ndarray:
    g = np.array([c == "G" for c in seq], dtype=float)
    c = np.array([c == "C" for c in seq], dtype=float)
    k = np.cumsum(np.concatenate([[0.0], g - c]))
    d = np.cumsum(np.concatenate([[0.0], g + c]))
    num = k[window:] - k[:-window]
    den = d[window:] - d[:-window]
    return np.where(den > 0, num / den, 0.0)


def bench_gc_skew(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    # synthetic genome: positive skew in first half, negative in second
    n = 400
    seq = []
    for i in range(n):
        if i < n // 2:
            seq.append("G" if rng.rand() < 0.4 else "ACGT"[rng.randint(4)])
        else:
            seq.append("C" if rng.rand() < 0.4 else "ACGT"[rng.randint(4)])
    s = gc_skew("".join(seq), 40)
    first = s[: len(s) // 2].mean()
    second = s[len(s) // 2 :].mean()
    return {"synthetic_skew_sign": float(first > 0 and second < 0)}
