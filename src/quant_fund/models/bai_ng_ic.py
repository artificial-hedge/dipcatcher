"""Bai-Ng (2002) factor-number information criteria.

References
----------
- Bai, J. & Ng, S. (2002). "Determining the Number of
  Factors in Approximate Factor Models." *Econometrica*
  70(1), 191-221.
- Ahn, S.C. & Horenstein, A.R. (2013). "Eigenvalue Ratio
  Test for the Number of Factors." *Econometrica* 81(3),
  1203-1227.
- Onatski, A. (2010). "Determining the Number of Factors
  from Empirical Distribution of Eigenvalues." *Review of
  Economics and Statistics* 92(4), 1004-1016.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
``X = F L' + e`` is (T, N); the estimated factor space of
size k is the first k principal components (``sqrt(T)`` x
eigenvectors of ``X'X``). Bai-Ng minimize

    PC_p(k) = V(k) + k * sigma2 * ((N + T) / (N T)) * log(N T / (N + T)),
    IC_p1(k) = log V(k) + k * ((N + T) / (N T)) * log(N T / (N + T)),
    IC_p2(k) = log V(k) + k * ((N + T) / (N T)) * log(min(N, T)),
    IC_p3(k) = log V(k) + k * (log(min(N, T)) / min(N, T)),

where ``V(k)`` is the mean squared residual of the
k-component PCA and ``sigma2 = V(k_max)``. Ahn-Horenstein
augment with the eigenvalue-ratio ``ER = argmax lam_k /
lam_{k+1}`` and growth-ratio ``GR`` variants on the same
spectrum. The bench plants a static factor panel with a
known r and independent idiosyncratic noise; all criteria
must recover r while a pure-noise panel returns 0 or 1.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _v_k(x: FloatArray, k: int, eigv: FloatArray) -> float:
    """Mean squared residual of the k-PC approximation."""
    n_t, n_n = x.shape
    # V(k) = (sum of eigenvalues beyond k) / (N T) with the
    # X'X/(NT) normalization convention.
    return float(np.sum(eigv[k:]) / (n_t * n_n))


def bai_ng_ic(
    x: FloatArray,
    k_max: int = 8,
) -> dict[str, float]:
    """Bai-Ng PC/IC criteria + Ahn-Horenstein ER/GR."""
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 2 or min(xx.shape) < 20 or not np.all(np.isfinite(xx)):
        raise ValueError("bad panel")
    if float(np.std(xx)) < 1e-12:
        raise ValueError("degenerate")
    n_t, n_n = xx.shape
    xc = xx - xx.mean(axis=0)
    xx_scale = xc / np.maximum(xc.std(axis=0), 1e-12)
    xtx = xx_scale.T @ xx_scale
    eigv = np.linalg.eigvalsh(xtx)[::-1]
    eigv = np.maximum(eigv, 0.0)
    k_max = int(min(k_max, min(n_t, n_n) - 1))
    v = np.array([_v_k(xx_scale, k, eigv) for k in range(k_max + 2)])
    sigma2 = max(v[k_max], 1e-30)
    nt_ = (n_n + n_t) / (n_n * n_t)
    pc = np.array(
        [v[k] + k * sigma2 * nt_ * np.log(n_n * n_t / (n_n + n_t)) for k in range(k_max + 1)]
    )
    ic1 = np.array(
        [
            np.log(max(v[k], 1e-30)) + k * nt_ * np.log(n_n * n_t / (n_n + n_t))
            for k in range(k_max + 1)
        ]
    )
    ic2 = np.array(
        [np.log(max(v[k], 1e-30)) + k * nt_ * np.log(min(n_n, n_t)) for k in range(k_max + 1)]
    )
    ic3 = np.array(
        [
            np.log(max(v[k], 1e-30)) + k * np.log(min(n_n, n_t)) / min(n_n, n_t)
            for k in range(k_max + 1)
        ]
    )
    # Ahn-Horenstein: eigenvalue ratio + growth ratio on
    # demeaned-data eigenvalues (k starts at 0 -> r index+1
    # convention; ER excludes the 0-factor candidate).
    er = eigv[:-1] / np.maximum(eigv[1:], 1e-30)
    gr_num = np.log(1.0 + eigv[:-1] / np.maximum(eigv[1:], 1e-30) / np.sum(eigv))
    gr_den = np.log(1.0 + eigv[1:] / np.maximum(np.r_[eigv[2:], 1e-30], 1e-30) / np.sum(eigv))
    gr = gr_num / np.maximum(gr_den, 1e-30)
    return {
        "r_pc1": float(np.argmin(pc)),
        "r_ic1": float(np.argmin(ic1)),
        "r_ic2": float(np.argmin(ic2)),
        "r_ic3": float(np.argmin(ic3)),
        "r_er": float(np.argmax(er[:k_max]) + 1),
        "r_gr": float(np.argmax(gr[:k_max]) + 1),
        "eig_top": float(eigv[0]),
        "eig_gap_at_r": float(eigv[0] / max(eigv[1], 1e-30)),
    }


def synth_factor(
    seed: int = 20261231 + 333,
    n_t: int = 120,
    n_n: int = 60,
    r: int = 3,
) -> tuple[FloatArray, FloatArray]:
    """SYNTHETIC r-factor panel vs pure noise panel."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n_t, r))
    load = rng.standard_normal((n_n, r))
    panel = f @ load.T + rng.standard_normal((n_t, n_n))
    noise = rng.standard_normal((n_t, n_n))
    return panel, noise


def bench_bai_ng(seed: int = 20261231 + 333) -> dict[str, float]:
    """Wave-57 self-check: criteria recover planted r."""
    panel, noise = synth_factor(seed=seed)
    res = bai_ng_ic(panel)
    res0 = bai_ng_ic(noise)
    votes = [res["r_ic1"], res["r_ic2"], res["r_ic3"], res["r_er"], res["r_gr"]]
    majority = float(np.median(votes))
    ok = majority == 3.0 and res["r_ic1"] == 3.0 and res["r_er"] == 3.0 and res0["r_er"] <= 2.0
    return {
        "synthetic_r_ic1": res["r_ic1"],
        "synthetic_r_ic2": res["r_ic2"],
        "synthetic_r_ic3": res["r_ic3"],
        "synthetic_r_er": res["r_er"],
        "synthetic_r_gr": res["r_gr"],
        "synthetic_r_er_null": res0["r_er"],
        "synthetic_score": float(ok),
    }
