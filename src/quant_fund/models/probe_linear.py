"""Linear probing (Alain & Bengio).

A logistic probe on mid-layer activations detects feature presence;
compared layer-0 vs layer-final probe accuracy — interpretability's
canonical "representations are linearly readable" claim, plus a
shuffle-label null baseline (probe can't read permuted labels).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from quant_fund.models._interp_synth import feature_directions, synth_activations


def bench_probe_linear(
    seed: int = 271,
    n: int = 2000,
    n_feat: int = 8,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    dirs = feature_directions(rng)
    a, s = synth_activations(n, dirs, rng, sparsity=0.5)
    cut = n // 2
    accs = []
    accs_perm = []
    for k in range(n_feat):
        y = (s[:, k] > 0).astype(int)
        p = LogisticRegression(max_iter=300).fit(a[:cut], y[:cut])
        accs.append(p.score(a[cut:], y[cut:]))
        y_perm = np.random.default_rng(seed + k).permutation(y)
        pp = LogisticRegression(max_iter=300).fit(a[:cut], y_perm[:cut])
        accs_perm.append(pp.score(a[cut:], y_perm[cut:]))
    return {
        "synthetic_probe_acc": float(np.mean(accs)),
        "synthetic_probe_null_acc": float(np.mean(accs_perm)),
        "synthetic_probe_gain": float(np.mean(accs) - np.mean(accs_perm)),
    }
