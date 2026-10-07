"""Gordon-Newell normalizing-constant evaluation of a closed network (SYNTHETIC).

Direct convolution of node weight functions g_i(k) = (V_i/mu_i)^k gives
G(N); the throughput is X(N) = G(N-1)/G(N) (up to visit-ratio scaling).
Bench: throughput curve vs the MVA-implied throughput and vs a
balanced-flow bound N/max_i(V_i/mu_i sum) as the sanity ceiling.
"""

import numpy as np

from quant_fund.models._queue_synth import CN_MU, CN_N, CN_VISIT


def _gn_constants(n_pop: int) -> np.ndarray:
    d = CN_VISIT / CN_MU
    g = np.array([1.0])
    for i in range(len(d)):
        g = np.convolve(g, d[i] ** np.arange(n_pop + 1))[: n_pop + 1]
    return g


def bench_gordon_newell(seed: int = 4405) -> dict[str, float]:
    del seed
    g = _gn_constants(CN_N)
    x_gn = g[:-1] / g[1:]  # X(n) = G(n-1)/G(n)
    # MVA throughput at each n: X = n / sum_i R_i, R_i = D_i(1+W_i(n-1))
    w = np.zeros(len(CN_MU))
    d = CN_VISIT / CN_MU
    x_mva = np.zeros(CN_N)
    for n in range(1, CN_N + 1):
        r = d * (1.0 + w)
        x_mva[n - 1] = n / float(r.sum())
        w = x_mva[n - 1] * r
    err = float(np.max(np.abs(x_gn - x_mva)))
    bound = CN_N / float(np.sum(CN_VISIT / CN_MU))
    return {
        "synthetic_gn_x_err": err,
        "synthetic_gn_x_n": float(x_gn[-1]),
        "synthetic_gn_mva_x": float(x_mva[-1]),
        "synthetic_gn_bound": float(bound),
        "synthetic_gn_log_g": float(np.log(g[-1])),
    }
