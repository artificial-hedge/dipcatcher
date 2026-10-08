"""Button-debounce FSM: output changes only after N stable samples (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 653


def debounce(samples: np.ndarray, n_stable: int = 4) -> np.ndarray:
    out = np.zeros(len(samples), dtype=int)
    state = 0
    count = 0
    for t, s in enumerate(samples):
        if s == state:
            count = 0
        else:
            count += 1
            if count >= n_stable:
                state = int(s)
                count = 0
        out[t] = state
    return out


def bench_debounce_fsm(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 40
    for _ in range(trials):
        n = 200
        clean = np.zeros(n, dtype=int)
        # blocky ground truth: long runs of 0/1
        pos = 0
        while pos < n:
            run = rng.randint(20, 60)
            clean[pos : pos + run] = rng.randint(2)
            pos += run
        noisy = clean.copy()
        # glitch bursts of 1-2 samples (below stability threshold)
        for _ in range(30):
            i = rng.randint(0, n - 2)
            noisy[i] = 1 - noisy[i]
        out = debounce(noisy, 4)
        # output must equal clean within tolerance (last few samples may lag)
        ok += float(np.mean(out == clean) > 0.85)
    return {"synthetic_debounce_clean": ok / trials}
