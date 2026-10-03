"""Markov chain entropy rate (wave 285).

Analytic H = sum_s pi_s * H(P_s.) for a 2-state chain vs simulated
block-entropy rate H(X_1..n)/n for large n.
"""

import numpy as np

_SEED = 20261231 + 794


def _analytic(p: float, q: float) -> float:
    # p = P(stay in 0), q = P(stay in 1); flip rates a = 1-p, b = 1-q
    a, b = 1 - p, 1 - q
    pi0 = b / (a + b)

    def h(x: float) -> float:
        return -x * np.log2(x) - (1 - x) * np.log2(1 - x) if 0 < x < 1 else 0.0

    return pi0 * h(p) + (1 - pi0) * h(q)


def _simulate(p: float, q: float, n: int, seed: int) -> float:
    rng = np.random.RandomState(seed)
    x = np.zeros(n, dtype=int)
    for i in range(1, n):
        stay = p if x[i - 1] == 0 else q
        x[i] = x[i - 1] if rng.rand() < stay else 1 - x[i - 1]
    # block entropy of pairs: H(X_t, X_{t+1}) minus unigram = per-symbol rate approx
    pairs = x[:-1] * 2 + x[1:]
    _, cnt = np.unique(pairs, return_counts=True)
    pp = cnt / cnt.sum()
    h2 = -np.sum(pp * np.log2(pp))
    _, c1 = np.unique(x, return_counts=True)
    p1 = c1 / c1.sum()
    h1 = -np.sum(p1 * np.log2(p1))
    return h2 - h1  # conditional entropy = entropy rate


def bench_markov_entropy(seed: int = _SEED) -> dict[str, float]:
    errs = [
        abs(_analytic(p, q) - _simulate(p, q, 60000, seed + i))
        for i, (p, q) in enumerate([(0.9, 0.8), (0.7, 0.6), (0.95, 0.5)])
    ]
    return {"synthetic_entropy_rate": float(max(errs) < 0.02)}
