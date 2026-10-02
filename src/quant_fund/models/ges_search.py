"""GES — greedy equivalence search (Chickering 2002, DAG-space variant):
hill-climb edge insertions/deletions/reversals scoring linear-Gaussian
BIC locally. SHD + skeleton F1 vs correlation baseline.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._cs_synth import corr_baseline, shd
from quant_fund.models._csl_synth import lingam_sem, skeleton_f1


def _bic_local(X: np.ndarray, j: int, parents: list[int], n: int) -> float:
    if not parents:
        rss = float((X[:, j] ** 2).sum())
        return float(0.5 * n * np.log(max(rss / n, 1e-9)) + 0.5 * np.log(n))
    A = np.column_stack([X[:, p] for p in parents] + [np.ones(n)])
    b = np.linalg.lstsq(A, X[:, j], rcond=None)[0]
    r = X[:, j] - A @ b
    rss = float((r**2).sum())
    return float(0.5 * n * np.log(max(rss / n, 1e-9)) + 0.5 * np.log(n) * (len(parents) + 1))


def _is_dag(B: np.ndarray) -> bool:
    # DFS cycle check
    d = B.shape[0]
    color = [0] * d

    def dfs(v: int) -> bool:
        color[v] = 1
        for w in np.nonzero(B[v])[0]:
            if color[w] == 1:
                return False
            if color[w] == 0 and not dfs(w):
                return False
        color[v] = 2
        return True

    return all(color[v] != 0 or dfs(v) for v in range(d))


def bench_ges_search(seed: int = 2819, trials: int = 4, d: int = 6) -> dict[str, float]:
    shds, shds_b, f1s = [], [], []
    for t in range(trials):
        X, B, _ = lingam_sem(seed + 11 * t, d=d, noise="gauss")
        n = len(X)
        parents: list[list[int]] = [[] for _ in range(d)]
        improved = True
        while improved:
            improved = False
            best_delta, best_op = 0.0, None
            for j in range(d):
                base = _bic_local(X, j, parents[j], n)
                for i in range(d):
                    if i == j:
                        continue
                    if i not in parents[j]:
                        cand = parents[j] + [i]
                        delta = base - _bic_local(X, j, cand, n)
                        Bh = np.zeros((d, d))
                        for k in range(d):
                            for p in parents[k]:
                                Bh[p, k] = 1
                        Bh[i, j] = 1
                        if delta > best_delta and _is_dag(Bh):
                            best_delta, best_op = delta, ("add", i, j)
                    else:
                        cand = [p for p in parents[j] if p != i]
                        delta = base - _bic_local(X, j, cand, n)
                        if delta > best_delta:
                            best_delta, best_op = delta, ("del", i, j)
            if best_op:
                op, i, j = best_op
                if op == "add":
                    parents[j].append(i)
                else:
                    parents[j].remove(i)
                improved = True
        B_hat = np.zeros((d, d))
        for j in range(d):
            for p in parents[j]:
                B_hat[p, j] = 1.0
        shds.append(shd(B, B_hat))
        shds_b.append(shd(B, corr_baseline(X, 7)))
        f1s.append(skeleton_f1(B, B_hat))
    return {
        "synthetic_ges_shd": float(np.mean(shds)),
        "synthetic_corr_shd": float(np.mean(shds_b)),
        "synthetic_ges_skel_f1": float(np.mean(f1s)),
        "torch_available": 0.0,
    }
