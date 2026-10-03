"""Mallows model averaging (MMA) — Hansen's weight choice.

Nested candidate least-squares models are combined with weights
on the unit simplex minimizing the Mallows criterion
C(w) = ‖ŷ(w) − y‖² + 2σ̂² Σ_m w_m k_m — an unbiased estimate of
out-of-sample prediction risk. Beats single-model selection when
the truth sits between candidates.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure prediction-risk reduction on
generated nested designs — never market evidence.

References:
- Hansen, B. E. (2007). Least squares model averaging.
  *Econometrica* 75, 1175-1189 — the Mallows criterion and
  optimality under nested models.
- Hansen, B. E. (2014). Model averaging, asymptotic risk, and
  regressor groups. *Quantitative Economics* 5.
- Mallows, C. L. (1973). Some comments on C_p. *Technometrics*
  15 — the unbiased risk estimate behind C_p.
- Ando, T., Li, K.-C. (2014). A model-averaging approach for
  high-dimensional regression. *JASA* 109 — weight-space
  enumeration vs continuous optimization.

Composition: pure numpy — nested-OLS fits, criterion surface on
a simplex grid + simplex-constrained refinement; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def mma_fit(
    y: FloatArray,
    x: FloatArray,
    max_models: int | None = None,
) -> dict[str, float]:
    """Mallows model averaging over nested OLS candidates.

    ``x`` columns are ordered by importance (the nested sequence
    1..k). Returns the optimal simplex weights, averaged fit
    risk vs full-model and best-single baselines."""
    yy = np.asarray(y, dtype=np.float64).ravel()
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 2 or xx.shape[0] != yy.size:
        raise ValueError("x rows must match y")
    n, k_full = xx.shape
    if n < 60 or k_full < 2 or k_full > 10:
        raise ValueError("n>=60, k in 2..10")
    if not np.all(np.isfinite(yy)) or not np.all(np.isfinite(xx)):
        raise ValueError("finite inputs required")
    if np.min(np.std(xx, axis=0)) < 1e-9:
        raise ValueError("each regressor must vary")
    if np.std(yy) < 1e-9:
        raise ValueError("y must vary")

    m_max = k_full if max_models is None else min(max_models, k_full)
    if m_max < 2:
        raise ValueError("need >=2 candidate models")

    # nested candidates: model m uses first m columns (+intercept)
    fits: list[FloatArray] = []
    ks: list[int] = []
    resid_var_full = 0.0
    for m in range(1, m_max + 1):
        w = np.column_stack([np.ones(n), xx[:, :m]])
        b = np.linalg.lstsq(w, yy, rcond=None)[0]
        fits.append(yy - w @ b)  # residuals
        ks.append(m + 1)
        if m == m_max:
            resid_var_full = float(np.sum(fits[-1] ** 2) / max(1, n - m - 1))
    R = np.column_stack(fits)  # residual matrix (n × M)
    M = len(ks)
    kvec = np.array(ks, dtype=np.float64)
    sig2 = resid_var_full

    # Mallows criterion on fitted values: C(w) = ‖Ŷw − y‖² + 2σ̂²w'k
    # Ŷ w = y − R w → ‖y − R w − y‖² = w' R' R w
    gram = R.T @ R

    # simplex grid search (coarse) then coordinate refinement
    best_w = np.ones(M) / M
    best_c = float("inf")
    grid_res = 8
    if M == 2:
        grids = np.linspace(0, 1, grid_res + 1)
        for g in grids:
            w = np.array([g, 1 - g])
            c = float(w @ gram @ w + 2 * sig2 * w @ kvec)
            if c < best_c:
                best_c, best_w = c, w
    else:
        # coordinate descent from several starts
        rng = np.random.default_rng(11)
        starts = [np.eye(M)[i] for i in range(M)]
        starts.append(np.ones(M) / M)
        starts.append(rng.dirichlet(np.ones(M)))
        for w0 in starts:
            w = np.asarray(w0, dtype=np.float64).copy()
            for _ in range(200):
                improved = False
                for i in range(M):
                    for j in range(M):
                        if i == j or w[i] < 1e-9:
                            continue
                        for d in (0.25, 0.5, 1.0):
                            wn = w.copy()
                            move = min(wn[i], d)
                            wn[i] -= move
                            wn[j] += move
                            c_new = float(wn @ gram @ wn + 2 * sig2 * wn @ kvec)
                            c_old = float(w @ gram @ w + 2 * sig2 * w @ kvec)
                            if c_new < c_old - 1e-12:
                                w = wn
                                improved = True
                                break
                if not improved:
                    break
            c = float(w @ gram @ w + 2 * sig2 * w @ kvec)
            if c < best_c:
                best_c, best_w = c, w

    # averaged fit & prediction-risk comparison (in-sample MSPE
    # proxy via Mallows-adjusted residuals)
    resid_mma = R @ best_w
    mspe_mma = float(np.mean(resid_mma**2))
    # best single model by Mallows C
    c_each = np.array([float(np.sum(fits[m] ** 2) + 2 * sig2 * ks[m]) for m in range(M)])
    best_single = int(np.argmin(c_each))
    mspe_single = float(np.mean(fits[best_single] ** 2))
    mspe_full = float(np.mean(fits[-1] ** 2))

    return {
        "n": float(n),
        "n_models": float(M),
        "w_top": float(np.max(best_w)),
        "w_argmax_model": float(int(np.argmax(best_w)) + 1),
        "w_entropy": float(-np.sum(best_w * np.log(np.clip(best_w, 1e-12, 1.0)))),
        "mspe_mma": mspe_mma,
        "mspe_single": mspe_single,
        "mspe_full": mspe_full,
        "mma_vs_single": float(1.0 - mspe_mma / max(mspe_single, 1e-300)),
        "mma_vs_full": float(1.0 - mspe_mma / max(mspe_full, 1e-300)),
    }


def synth_nested(
    n: int = 500,
    decay: float = 0.55,
    k: int = 6,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Nested true model: y = Σ_j β_j x_j + e with decaying
    coefficients β_j = decay^j — mid-size models are competitive."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (n, k))
    betas = np.array([decay**j for j in range(k)]) * 1.5
    y = x @ betas + rng.normal(0.0, 1.0, n)
    return {"y": y, "x": x, "betas": betas}


def bench_model_averaging(seed: int = 20261231 + 238) -> dict[str, float]:
    """MMA self-check: weight spread tracks true model size and
    prediction risk stays at or below single-model choice.
    All ``synthetic_*``."""
    d = synth_nested(decay=0.45, seed=seed)
    out = mma_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    # sharp-truth contrast: β₁ only → MMA should pick smaller models
    d1 = synth_nested(decay=0.01, seed=seed + 1)
    out1 = mma_fit(np.asarray(d1["y"]), np.asarray(d1["x"]))
    out_b = mma_fit(np.asarray(d["y"]), np.asarray(d["x"]))

    w_top = float(out["w_top"])
    am = float(out["w_argmax_model"])
    am1 = float(out1["w_argmax_model"])
    return {
        "synthetic_w_top": w_top,
        "synthetic_w_entropy": float(out["w_entropy"]),
        "synthetic_argmax_model": am,
        "synthetic_argmax_sharp": am1,
        "synthetic_mspe_mma": float(out["mspe_mma"]),
        "synthetic_mspe_single": float(out["mspe_single"]),
        "synthetic_mma_vs_single": float(out["mma_vs_single"]),
        "synthetic_mma_vs_full": float(out["mma_vs_full"]),
        "synthetic_detects": float(
            float(out["mspe_mma"]) <= float(out["mspe_single"]) * 1.05
            and am > am1
            and float(out["w_entropy"]) > 0.05
        ),
        "synthetic_determinism": float(w_top == float(out_b["w_top"])),
    }
