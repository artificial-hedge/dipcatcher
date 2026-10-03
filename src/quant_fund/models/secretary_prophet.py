"""Secretary problem + prophet inequality on a fixed reward sequence.

Secretary: observe first n/e, then pick first exceeding the sample
max. Prophet: threshold rule at the median of the max distribution.
Bench reports achieved reward vs hindsight max on shuffled draws.
"""

import numpy as np


def _seq(rng: np.random.Generator, n: int = 24) -> np.ndarray:
    return rng.gamma(2.0, 1.0, size=n)


def bench_secretary_prophet(seed: int = 5409) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n, trials = 24, 2000
    sec_ok = 0
    proph_ratio = 0.0
    for _ in range(trials):
        x = _seq(rng, n)
        h = max(x)
        # secretary
        k = max(1, int(n / np.e))
        thr = max(x[:k])
        pick = next((v for v in x[k:] if v > thr), x[-1])
        sec_ok += pick == h
        # prophet: threshold at E[max]/2 style -> use mean-max/2
        t = 0.5 * np.mean([_seq(rng, n).max() for _ in range(64)])
        got = next((v for v in x if v >= t), x[-1])
        proph_ratio += got / h
    return {
        "synthetic_sec_hit": sec_ok / trials,
        "synthetic_sec_bound": 1.0 / np.e,
        "synthetic_prophet_ratio": proph_ratio / trials,
        "synthetic_prophet_bound": 0.5,
    }
