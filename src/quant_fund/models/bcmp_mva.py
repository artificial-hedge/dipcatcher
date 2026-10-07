"""Closed BCMP network via mean-value analysis (Reiser-Lavenberg) (SYNTHETIC).

Iterate over population n = 1..N: residence time
R_i(n) = D_i * (1 + W_i(n-1)) with service demand D_i = V_i/mu_i, throughput
X(n) = n / sum_i R_i(n), mean queue W_i(n) = X(n) R_i(n). Bench: per-node
mean queue lengths vs the Gordon-Newell normalizing-constant exact values.
"""

import numpy as np

from quant_fund.models._queue_synth import CN_MU, CN_N, CN_VISIT


def mva_queues(n_pop: int) -> np.ndarray:
    w = np.zeros(len(CN_MU))
    d = CN_VISIT / CN_MU  # service demands
    for n in range(1, n_pop + 1):
        r = d * (1.0 + w)
        x = n / float(r.sum())
        w = x * r
    return w


def gordon_newell_queues(n_pop: int) -> np.ndarray:
    """Exact mean queue lengths from the GN normalizing constant."""
    # G(n) via convolution of node contributions g_i(k) = (V_i/mu_i)^k
    d = CN_VISIT / CN_MU
    g = np.array([1.0])
    for i in range(len(d)):
        gi = d[i] ** np.arange(n_pop + 1)
        g = np.convolve(g, gi)[: n_pop + 1]
    g_n = g[n_pop]
    # E[N_i] = sum_{k>=1} P(N_i >= k) = sum_k d_i^k G(N-k) / G(N)
    means = np.zeros(len(d))
    for i in range(len(d)):
        k = np.arange(1, n_pop + 1)
        means[i] = float(np.sum(d[i] ** k * g[n_pop - k])) / g_n
    return means


def bench_bcmp_mva(seed: int = 4403) -> dict[str, float]:
    del seed
    w_mva = mva_queues(CN_N)
    w_exact = gordon_newell_queues(CN_N)
    err = float(np.max(np.abs(w_mva - w_exact)))
    naive = np.full(len(w_mva), CN_N / len(w_mva))
    return {
        "synthetic_mva_err": err,
        "synthetic_mva_w0": float(w_mva[0]),
        "synthetic_mva_exact_w0": float(w_exact[0]),
        "synthetic_mva_uniform_err": float(np.max(np.abs(naive - w_exact))),
        "synthetic_mva_total": float(w_mva.sum()),
    }
