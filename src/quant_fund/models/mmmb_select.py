"""MMPC/Markov-blanket selection (Tsamardinos et al. 2003) — for each (SYNTHETIC)
variable, grow its parent-children set by max-min dependence then add
spouses; symmetrized union gives the skeleton. F1 vs correlation.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._csl_synth import lingam_sem, skeleton_f1
from quant_fund.models.fci_alg import _partial_corr


def _mb(X: np.ndarray, t: int, alpha: float = 0.1) -> set[int]:
    d = X.shape[1]
    pc: set[int] = set()
    changed = True
    while changed:
        changed = False
        # add var maximizing min |partial corr| given PC
        best_j, best_dep = -1, alpha
        for j in range(d):
            if j == t or j in pc:
                continue
            dep = abs(_partial_corr(X, t, j, list(pc)))
            if dep > best_dep:
                best_dep, best_j = dep, j
        if best_j >= 0:
            pc.add(best_j)
            changed = True
        # remove vars now independent
        for j in list(pc):
            if abs(_partial_corr(X, t, j, [k for k in pc if k != j])) < alpha:
                pc.discard(j)
                changed = True
    return pc


def bench_mmmb_select(seed: int = 2831, trials: int = 4, d: int = 6) -> dict[str, float]:
    f1s = []
    for t in range(trials):
        X, B, _ = lingam_sem(seed + 17 * t, d=d, noise="gauss")
        A = np.zeros((d, d))
        for v in range(d):
            for j in _mb(X, v):
                A[v, j] = A[j, v] = 1.0
        f1s.append(skeleton_f1(B, A))
    return {"synthetic_mmmb_skel_f1": float(np.mean(f1s)), "synthetic_torch_available": 0.0}
