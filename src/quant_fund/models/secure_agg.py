"""Secure aggregation — pairwise mask cancellation (Bonawitz 2017) (SYNTHETIC).

Clients add pairwise-secret masks to their updates; the server sums
masked updates and masks cancel — the aggregate is exact while each
client's update is hidden. Verify masked-sum == plain-sum.
"""

from __future__ import annotations

import numpy as np


def bench_secure_agg(seed: int = 421, n_clients: int = 6, dim: int = 20) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    updates = rng.normal(0, 1, (n_clients, dim))
    # pairwise masks: mask_ij = -mask_ij for i<j pairs, drawn once per pair
    masked = updates.copy()
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            m = rng.normal(0, 100, dim)
            masked[i] += m
            masked[j] -= m
    plain = updates.sum(0)
    secure = masked.sum(0)
    err = float(np.abs(plain - secure).max())
    # each masked update reveals nothing (correlation with true update ≈ 0)
    leak = float(np.corrcoef(masked[0], updates[0])[0, 1])
    return {
        "synthetic_sagg_sum_err": err,
        "synthetic_sagg_masked_true_corr": leak,
        "synthetic_sagg_exact": float(err < 1e-8),
    }
