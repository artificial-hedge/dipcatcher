"""Circuit ablation / mean-ablation (Olah lab conventions).

Zeroing the activation mass along feature direction k degrades the
probe for k but leaves other features readable — the circuit-surgery
sanity check that the direction carries the feature's causal load.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from quant_fund.models._interp_synth import feature_directions, synth_activations


def bench_circuit_ablation(
    seed: int = 283,
    n: int = 2000,
    target: int = 0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    dirs = feature_directions(rng)
    a, s = synth_activations(n, dirs, rng, sparsity=0.5)
    cut = n // 2
    v = dirs[target]
    a_abl = a - (a @ v[:, None]) * v[None, :]
    y = (s[:, target] > 0).astype(int)
    p_before = LogisticRegression(max_iter=300).fit(a[:cut], y[:cut]).score(a[cut:], y[cut:])
    p_after = LogisticRegression(max_iter=300).fit(a_abl[:cut], y[:cut]).score(a_abl[cut:], y[cut:])
    # collateral: probe a different feature on ablated activations
    k2 = 1
    y2 = (s[:, k2] > 0).astype(int)
    q_before = LogisticRegression(max_iter=300).fit(a[:cut], y2[:cut]).score(a[cut:], y2[cut:])
    q_after = (
        LogisticRegression(max_iter=300).fit(a_abl[:cut], y2[:cut]).score(a_abl[cut:], y2[cut:])
    )
    return {
        "synthetic_abl_target_drop": float(p_before - p_after),
        "synthetic_abl_collateral_drop": float(q_before - q_after),
        "synthetic_abl_selectivity": float((p_before - p_after) - (q_before - q_after)),
    }
