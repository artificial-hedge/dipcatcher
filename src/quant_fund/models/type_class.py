"""Method of types (wave 285).

For iid ternary source, empirical type T has probability
Pr(T) = multinomial coeff * prod p^x — verified against the exact
multinomial oracle on small n.
"""

import math

import numpy as np

_SEED = 20261231 + 797


def _prob_type(p: np.ndarray, counts: np.ndarray) -> float:
    n = counts.sum()
    coeff = math.factorial(n)
    for c in counts:
        coeff //= math.factorial(int(c))
    return float(coeff * np.prod(p**counts))


def empirical_prob(p: np.ndarray, counts: np.ndarray) -> float:
    return _prob_type(p, counts)


def bench_type_class(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    p = np.array([0.5, 0.3, 0.2])
    # all type probabilities must sum to 1 over compositions of n=8
    n = 8
    tot = 0.0
    for a in range(n + 1):
        for b in range(n - a + 1):
            c = n - a - b
            tot += empirical_prob(p, np.array([a, b, c]))
    # simulated: draw sequence, check empirical type freq matches
    seq = rng.choice(3, 40000, p=p).reshape(-1, 8)
    types = (seq == 0).sum(1) * 100 + (seq == 1).sum(1) * 10 + (seq == 2).sum(1)
    u, cnt = np.unique(types, return_counts=True)
    freq = dict(zip(u, cnt / cnt.sum(), strict=True))
    # oracle freq for a common type (4,2,2)
    want = _prob_type(p, np.array([4, 2, 2]))
    got = freq.get(4 * 100 + 2 * 10 + 2, 0.0)
    return {"synthetic_type_prob": float(abs(tot - 1.0) < 1e-9 and abs(got - want) / want < 0.1)}
