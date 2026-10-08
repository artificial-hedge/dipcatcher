"""Miller-Madow entropy estimator (wave 285) (SYNTHETIC).

Plug-in H is biased low on small samples; Miller-Madow adds
(m-1)/(2n) nats correction, shrinking bias vs the true entropy.
"""

import numpy as np

_SEED = 20261231 + 799


def _plug(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def miller_madow(counts: np.ndarray, k: int) -> float:
    m = int(np.count_nonzero(counts))
    return float(_plug(counts) + (m - 1) / (2.0 * counts.sum()))


def bench_miller_madow(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    k = 10
    truth = np.log(k)
    errs_plug, errs_mm = [], []
    for _ in range(40):
        cnt = np.bincount(rng.randint(0, k, 30), minlength=k)
        errs_plug.append(abs(_plug(cnt) - truth))
        errs_mm.append(abs(miller_madow(cnt, k) - truth))
    return {"synthetic_miller_madow": float(np.mean(errs_mm) < 0.8 * np.mean(errs_plug))}
