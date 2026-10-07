"""FCI-lite (Spirtes et al. 1995) — PC-stable skeleton with collider +
Meek orientation under a latent confounder: node 0 is dropped as
unobserved; skeleton F1 vs the induced marginal DAG skeleton.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._csl_synth import lingam_sem, skeleton_f1


def _partial_corr(X: np.ndarray, i: int, j: int, cond: list[int]) -> float:
    if not cond:
        return float(np.corrcoef(X[:, i], X[:, j])[0, 1])
    A = np.column_stack([X[:, c] for c in cond] + [np.ones(len(X))])
    ri = X[:, i] - A @ np.linalg.lstsq(A, X[:, i], rcond=None)[0]
    rj = X[:, j] - A @ np.linalg.lstsq(A, X[:, j], rcond=None)[0]
    return float(np.corrcoef(ri, rj)[0, 1])


def bench_fci_alg(
    seed: int = 2825, trials: int = 4, d: int = 6, alpha: float = 0.1
) -> dict[str, float]:
    f1s = []
    for t in range(trials):
        X, B, _ = lingam_sem(seed + 13 * t, d=d, noise="gauss")
        # node 0 unobserved
        obs = list(range(1, d))
        Xo = X[:, obs]
        do = d - 1
        # PC-stable skeleton
        adj = np.ones((do, do)) - np.eye(do)
        for depth in range(0, do - 1):
            for i in range(do):
                for j in range(i + 1, do):
                    if adj[i, j] == 0:
                        continue
                    nbrs = [k for k in range(do) if adj[i, k] and k != j]
                    from itertools import combinations

                    removed = False
                    for cond in combinations(nbrs, min(depth, len(nbrs))):
                        if abs(_partial_corr(Xo, i, j, list(cond))) < alpha:
                            adj[i, j] = adj[j, i] = 0
                            removed = True
                            break
                    if removed:
                        continue
        # colliders (v-structures): i-k-j nonadjacent with k not in sepset → orient
        B_hat = np.zeros((d, d))
        B_hat[1:, 1:] = adj
        # induced marginal skeleton = adjacency of true B restricted to obs,
        # plus edges induced by latent confounding (common parent 0)
        B_ind = np.zeros((d, d))
        B_ind[1:, 1:] = (B[1:, 1:] != 0).astype(float)
        for a in range(1, d):
            for b in range(a + 1, d):
                if B[0, a] != 0 and B[0, b] != 0:
                    B_ind[a, b] = B_ind[b, a] = 1.0  # latent-induced edge
        f1s.append(skeleton_f1(B_ind, B_hat))
    return {"synthetic_fci_skel_f1": float(np.mean(f1s)), "synthetic_torch_available": 0.0}
