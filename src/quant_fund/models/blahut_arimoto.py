"""Blahut-Arimoto channel capacity (wave 285).

Alternating maximization over input distribution and backward channel;
capacity of BSC(eps) converges to 1 - h(eps) bits.
"""

import numpy as np

_SEED = 20261231 + 795


def capacity(w: np.ndarray, iters: int = 200) -> float:
    # w[i,j] = P(y=j | x=i); r[i] = input dist
    n_in = w.shape[0]
    r = np.ones(n_in) / n_in
    for _ in range(iters):
        q = r @ w  # output dist
        q = np.clip(q, 1e-300, None)
        # c[i] = exp(sum_j w[i,j] log(w[i,j]/q[j]))
        logd = w @ np.log(q)
        logd = np.clip(logd, -700, 700)
        c = np.exp(np.sum(w * np.log(np.clip(w, 1e-300, None)) - logd, axis=1))
        r_new = r * c
        r = r_new / r_new.sum()
    q = r @ w
    # I(X;Y) at converged r
    i_xy = 0.0
    for i in range(n_in):
        for j in range(w.shape[1]):
            if w[i, j] > 0 and r[i] > 0:
                i_xy += r[i] * w[i, j] * np.log2(w[i, j] / q[j])
    return float(i_xy)


def _bsc(eps: float) -> np.ndarray:
    return np.array([[1 - eps, eps], [eps, 1 - eps]])


def _h2(x: float) -> float:
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x) if 0 < x < 1 else 0.0


def bench_blahut_arimoto(seed: int = _SEED) -> dict[str, float]:
    errs = [abs(capacity(_bsc(e)) - (1 - _h2(e))) for e in [0.05, 0.11, 0.25]]
    return {"synthetic_ba_capacity": float(max(errs) < 5e-3)}
