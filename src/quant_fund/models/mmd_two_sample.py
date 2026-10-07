"""MMD two-sample test (Gretton et al. 2012): unbiased U-statistic with (SYNTHETIC)
median-heuristic Gaussian kernel; permutation p-value. Correctly
accepts H0 at shift 0, rejects at shift 0.6.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._it_synth import med_bw, rbf, twosample


def _mmd_u(X: np.ndarray, Y: np.ndarray, bw: float) -> float:
    K = rbf(X, X, bw)
    L = rbf(Y, Y, bw)
    P = rbf(X, Y, bw)
    n, m = len(X), len(Y)
    np.fill_diagonal(K, 0)
    np.fill_diagonal(L, 0)
    return float(K.sum() / (n * (n - 1)) + L.sum() / (m * (m - 1)) - 2 * P.mean())


def bench_mmd_two_sample(seed: int = 2873, perms: int = 200) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    X0, Y0 = twosample(seed, shift=0.0)
    X1, Y1 = twosample(seed + 1, shift=0.6)
    pvals = {}
    for tag, (X, Y) in {"null": (X0, Y0), "alt": (X1, Y1)}.items():
        bw = med_bw(X, Y)
        stat = _mmd_u(X, Y, bw)
        Z = np.vstack([X, Y])
        n = len(X)
        ge = 0
        for _ in range(perms):
            perm = rng.permutation(len(Z))
            if _mmd_u(Z[perm[:n]], Z[perm[n:]], bw) >= stat:
                ge += 1
        pvals[tag] = (ge + 1) / (perms + 1)
    return {
        "synthetic_mmd_pval_null": float(pvals["null"]),
        "synthetic_mmd_pval_alt": float(pvals["alt"]),
        "synthetic_mmd_sep": float(pvals["null"] - pvals["alt"]),
        "synthetic_torch_available": 0.0,
    }
