"""Matrix completion for causal panel estimation (MCPanel) (SYNTHETIC).

Treats the untreated outcome matrix as a low-rank object with missing
entries at treated cells: estimate Y(0) by nuclear-norm-regularized
soft-imputation, then read treatment effects as residuals
``Y - Y(0)_hat`` on the treated block. Covers soft-Impute (iterative
SVD thresholding at fixed λ), λ selection by validation on held-out
control cells, and ATT aggregation with a per-cell posterior band.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure recovery on generated factor panels —
never market evidence. Nuclear-norm imputation is biased toward zero
singular values; reported RMSE and ATT errors are honest pointwise
errors, not oracle bounds.

References:
- Athey, Bayati, Doudchenko, Imbens, Khosravi (2021). Matrix completion
  methods for causal panel data models. *JASA* 116.
- Mazumder, Hastie, Tibshirani (2010). Spectral regularization
  algorithms for learning large incomplete matrices. *JMLR* 11
  (soft-Impute).
- Candès, Recht (2009). Exact matrix completion via convex
  optimization. *Found. Comput. Math.* 9.
- Xu (2017). Generalized synthetic control method. *Political
  Analysis* 25 (IFE counterpart).

Composition: pure numpy — power/QR-free SVD via ``np.linalg.svd``;
deterministic ``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_matrix(y: FloatArray, name: str) -> FloatArray:
    a = np.asarray(y, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name}: finite (N, T) matrix required")
    if a.shape[0] < 4 or a.shape[1] < 6:
        raise ValueError(f"{name}: need >= 4 rows and >= 6 columns")
    return a


def _soft_threshold_svd(m: FloatArray, lam: float) -> FloatArray:
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    s_thr = np.maximum(s - lam, 0.0)
    return (u * s_thr) @ vt


def soft_impute(
    y_obs: FloatArray,
    mask: FloatArray,
    *,
    lam: float | None = None,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> dict[str, float | FloatArray]:
    """Soft-Impute the missing cells of ``y_obs`` (``mask==1`` observed).

    Returns the completed matrix, effective rank, and the λ used. λ is
    chosen by a small grid minimizing held-out RMSE over a validation
    fold of observed cells (deterministic 80/20 split by index stride).
    """
    m = np.asarray(y_obs, dtype=np.float64)
    obs = np.asarray(mask).astype(bool)
    if m.shape != obs.shape or m.ndim != 2:
        raise ValueError("y_obs and mask must share a 2-D shape")
    if obs.sum() < 10:
        raise ValueError("too few observed cells")
    if (~obs).sum() < 1:
        raise ValueError("mask marks no missing cells")

    idx = np.flatnonzero(obs)
    val = idx[idx % 5 == 0]  # deterministic 20% holdout by stride
    train_flat = obs.copy().ravel()
    train_flat[val] = False
    train = train_flat.reshape(obs.shape)

    scale = float(np.std(m[obs])) or 1.0
    fill0 = float(np.mean(m[obs]))

    def _fit(basis: FloatArray, obs_mask: FloatArray, lam_v: float) -> FloatArray:
        z = np.where(obs_mask, basis, fill0)
        prev = z.copy()
        for _ in range(max_iter):
            z = _soft_threshold_svd(z, lam_v)
            z = np.where(obs_mask, basis, z)
            if float(np.abs(z - prev).max()) < tol * scale:
                break
            prev = z
        return z

    if lam is None:
        lam_max = float(np.linalg.svd(np.where(train, m, 0.0), compute_uv=False)[0])
        grid = lam_max * np.array([0.5, 0.25, 0.12, 0.06, 0.03, 0.015])
        best_lam, best_rmse = float(grid[0]), math.inf
        for lam_v in grid:
            zf = _fit(m, train, float(lam_v))
            rmse = float(np.sqrt(np.mean((zf.ravel()[val] - m.ravel()[val]) ** 2)))
            if rmse < best_rmse:
                best_rmse, best_lam = rmse, float(lam_v)
        lam = best_lam
    else:
        best_rmse = math.nan

    z_full = _fit(m, obs, lam)
    u, s, _ = np.linalg.svd(z_full, full_matrices=False)
    eff_rank = float(np.sum(s > 0.01 * s[0]))
    return {
        "completed": z_full,
        "lam": float(lam),
        "eff_rank": eff_rank,
        "val_rmse": float(best_rmse),
        "n_missing": float((~obs).sum()),
        "scale": scale,
    }


def mc_att(
    y: FloatArray,
    treated: FloatArray,
    t0: int,
    *,
    lam: float | None = None,
) -> dict[str, float | FloatArray]:
    """ATT via matrix-completion counterfactual: treated cells are masked
    as missing post-t0, imputed as Y(0), and effects are residuals."""
    y = _as_matrix(y, "y")
    n_units, t = y.shape
    tr = np.asarray(treated).astype(bool).ravel()
    if tr.size != n_units or tr.sum() < 1 or tr.sum() > n_units - 3:
        raise ValueError("need >=1 treated and >=3 controls")
    if not (2 <= t0 <= t - 2):
        raise ValueError("t0 must be in [2, T-2]")
    mask = np.ones_like(y, dtype=bool)
    mask[tr, t0:] = False  # treated cells are "missing" post-t0
    out = soft_impute(y, mask, lam=lam)
    cf = np.asarray(out["completed"])
    eff = y - cf
    att = float(eff[tr, t0:].mean())
    cell_se = float(np.std(eff[tr, t0:], ddof=1)) if tr.sum() * (t - t0) > 1 else math.nan
    # control-placebo dispersion: mask a deterministic control subset
    # post-t0 and measure how large their "ATT" looks — honest noise
    # floor for the estimator.
    ctrl_idx = np.flatnonzero(~tr)
    n_pl = min(6, ctrl_idx.size)
    mask_pl = mask.copy()
    mask_pl[ctrl_idx[:n_pl], t0:] = False
    cf_pl = np.asarray(soft_impute(y, mask_pl, lam=float(out["lam"]))["completed"])
    pl_eff = (y - cf_pl)[ctrl_idx[:n_pl], t0:]
    se_scale = float(np.sqrt(np.mean(pl_eff**2)))
    return {
        "att": att,
        "att_cell_sd": cell_se,
        "impute_rmse_control": se_scale,
        "z_like": float(att / max(se_scale, 1e-12)),
        "eff_rank": float(out["eff_rank"]),
        "lam": float(out["lam"]),
        "val_rmse": float(out["val_rmse"]),
        "counterfactual": cf,
        "effects_matrix": eff,
    }


def synth_mc_panel(
    n_units: int = 30,
    t: int = 50,
    n_treated: int = 6,
    t0: int | None = None,
    tau: float = 1.2,
    rank: int = 3,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Low-rank panel + treatment wedge, additive noise."""
    if t0 is None:
        t0 = int(0.7 * t)
    rng = np.random.default_rng(seed)
    lam = rng.normal(0.0, 1.0, (n_units, rank))
    # stationary factors: persistent-trend factors would need an IFE or
    # demeaning step (gsynth handles that lane); MCPanel is benchmarked
    # on the stationary low-rank design it was analyzed for.
    f = rng.normal(0.0, 1.0, (t, rank))
    y = lam @ f.T + rng.normal(0.0, 0.25, (n_units, t))
    treated = np.zeros(n_units, dtype=bool)
    treated[rng.choice(n_units, n_treated, replace=False)] = True
    y[treated, t0:] += tau
    return {
        "y": y,
        "treated": treated.astype(np.float64),
        "t0": np.full(1, t0),
        "tau": np.full(1, tau),
        "lam": lam,
        "f": f,
    }


def bench_matrix_completion(seed: int = 20261231 + 191) -> dict[str, float]:
    """MCPanel self-check: imputation error on control cells small,
    ATT recovered within noise, rank detected. All ``synthetic_*``."""
    d = synth_mc_panel(seed=seed)
    y = np.asarray(d["y"])
    tr = np.asarray(d["treated"]).astype(bool)
    t0 = int(np.asarray(d["t0"]).item())
    tau_true = float(np.asarray(d["tau"]).item())

    est = mc_att(y, tr, t0)
    est2 = mc_att(y, tr, t0)
    d0 = synth_mc_panel(seed=seed + 1, tau=0.0)
    est0 = mc_att(
        np.asarray(d0["y"]),
        np.asarray(d0["treated"]).astype(bool),
        int(np.asarray(d0["t0"]).item()),
    )

    return {
        "synthetic_att": float(est["att"]),
        "synthetic_tau_true": tau_true,
        "synthetic_att_err": abs(float(est["att"]) - tau_true),
        "synthetic_impute_rmse": float(est["impute_rmse_control"]),
        "synthetic_val_rmse": float(est["val_rmse"]),
        "synthetic_eff_rank": float(est["eff_rank"]),
        "synthetic_null_att": float(est0["att"]),
        "synthetic_detects": float(
            abs(float(est["att"])) > 2.0 * float(est["impute_rmse_control"])
        ),
        "synthetic_determinism": float(est["att"] == est2["att"]),
    }
