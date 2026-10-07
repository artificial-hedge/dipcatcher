"""Independence Metropolis–Hastings with a Student-t envelope for a (SYNTHETIC)
heavy-tail target: correct tail coverage where a random-walk proposal
misses rare regions. Metric = tail-quantile error + ESS.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t


def bench_indep_mh(seed: int = 2987, n: int = 4000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    # target: 1-D Student-t df=4 scaled 1.5 (tails ~ t)
    df, scale = 4.0, 1.5

    def lt(x: float) -> float:
        return float(student_t.logpdf(x, df, scale=scale))

    def lq(x: float) -> float:
        return float(student_t.logpdf(x, df, scale=scale * 1.3))

    x = 0.0
    samples = []
    acc = 0
    for _ in range(n):
        y = student_t.rvs(df, scale=scale * 1.3, random_state=rng)
        if np.log(rng.uniform()) < lt(y) - lt(x) + lq(x) - lq(y):
            x = float(y)
            acc += 1
        samples.append(x)
    s = np.asarray(samples[500:])
    q_true = np.quantile(
        student_t.rvs(df, scale=scale, size=20000, random_state=0), [0.01, 0.5, 0.99]
    )
    q_hat = np.quantile(s, [0.01, 0.5, 0.99])
    tail_err = float(np.abs(q_hat - q_true).mean())
    return {
        "synthetic_imh_tail_err": tail_err,
        "synthetic_imh_accept": float(acc / n),
        "synthetic_torch_available": 0.0,
    }
