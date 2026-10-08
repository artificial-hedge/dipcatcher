"""Coverage-guided mutational fuzzer on a toy branchy target (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 670


def _target_cov(x: list[int]) -> int:
    """Branchy toy: coverage = distinct branch ids hit."""
    cov = set()
    v = 0
    for i, t in enumerate(x):
        if t > 10:
            cov.add(1)
        if t < -5:
            cov.add(2)
        v += t * (i + 1)
    if v > 100:
        cov.add(3)
    if -50 < v < 0:
        cov.add(4)
    if len(set(x)) > 5:
        cov.add(5)
    return len(cov)


def bench_fuzzer_mutate(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    covers = []
    for _ in range(20):
        corpus = [[0] * 8]
        best = 0
        for _ in range(300):
            i = int(rng.randint(len(corpus)))
            x = list(corpus[i])
            # mutation: perturb a few entries
            for _ in range(rng.randint(1, 3)):
                x[int(rng.randint(8))] += int(rng.randint(-15, 16))
            c = _target_cov(x)
            if c > best:
                best = c
                corpus.append(x)
        covers.append(best)
    return {"synthetic_fuzz_coverage": float(np.mean(covers)) / 5.0}
