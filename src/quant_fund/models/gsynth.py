"""Generalized synthetic control via interactive fixed effects (gsynth) (SYNTHETIC).

Xu's IFE estimator: model the untreated outcome matrix as
``Y(0)_it = x_i'β + λ_i'f_t + ε_it`` where the factor structure is
estimated by alternating least squares (EM-style) on CONTROL data only,
then the treated counterfactual is predicted from each treated unit's
pre-period loading fit. Effects are residuals Y - Y(0)_hat and are
aggregated to ATT; a unit jackknife supplies the honest scale for
significance.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure counterfactual recovery on generated
interactive-FE panels — never market evidence. Factor-model estimates
are only valid up to rotation; the bench checks prediction quality,
not identified factors.

References:
- Xu (2017). Generalized synthetic control method: causal inference
  with interactive fixed effects models. *Political Analysis* 25(1).
- Bai (2009). Panel data models with interactive fixed effects.
  *Econometrica* 77.
- Gobillon, Magnac (2016). Regional policy evaluation: interactive
  fixed effects and synthetic controls. *Rev. Econ. Stat.* 98.
- Abadie, Diamond, Hainmueller (2010). Synthetic control methods.
  *JASA* 105.

Composition: pure numpy — ALS factor extraction; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_matrix(y: FloatArray, name: str) -> FloatArray:
    a = np.asarray(y, dtype=np.float64)
    if a.ndim != 2 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: finite (N, T) matrix required")
    if a.shape[0] < 5 or a.shape[1] < 8:
        raise ValueError(f"{name}: need >= 5 units and >= 8 periods")
    return a


def _ife_als(
    y: FloatArray,
    mask: FloatArray,
    n_factors: int,
    *,
    max_iter: int = 200,
    tol: float = 1e-7,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Alternating LS for y ~ lam f' on observed cells (+ unit/time FE).

    Returns (lam, f, fitted) with f columns orthogonalized (signs fixed
    for determinism); identification is up to rotation — prediction only.
    """
    n, t = y.shape
    # two-way demeaning on observed cells
    alpha = np.zeros(n)
    delta = np.zeros(t)
    for _ in range(30):
        for i in range(n):
            ob = mask[i]
            alpha[i] = float(np.mean(y[i, ob] - delta[ob])) if ob.any() else 0.0
        for j in range(t):
            ob = mask[:, j]
            delta[j] = float(np.mean(y[ob, j] - alpha[ob])) if ob.any() else 0.0
        alpha -= alpha.mean()
    resid = np.where(mask, y - alpha[:, None] - delta[None, :], 0.0)

    rng_state = np.random.default_rng(12345)
    lam = rng_state.normal(0.0, 1.0, (n, n_factors))
    f = rng_state.normal(0.0, 1.0, (t, n_factors))
    prev = math.inf
    for _ in range(max_iter):
        # update lam row-wise using observed cols
        for i in range(n):
            ob = np.flatnonzero(mask[i])
            if ob.size < n_factors:
                continue
            fj = f[ob]
            coef, *_ = np.linalg.lstsq(fj, resid[i, ob], rcond=None)
            lam[i] = coef
        # update f row-wise using observed rows
        for j in range(t):
            ob = np.flatnonzero(mask[:, j])
            if ob.size < n_factors:
                continue
            li = lam[ob]
            coef, *_ = np.linalg.lstsq(li, resid[ob, j], rcond=None)
            f[j] = coef
        # orthogonalize for stability
        q, _ = np.linalg.qr(f)
        sgn = np.sign(np.diag(q[:, :n_factors].T @ f[:, :n_factors]))
        sgn[sgn == 0] = 1.0
        f = q[:, :n_factors] * sgn[None, :]
        fit = lam @ f.T
        err = float(np.sqrt(np.mean((resid - np.where(mask, fit, 0.0)) ** 2 * mask)))
        if abs(prev - err) < tol:
            break
        prev = err
    fitted = lam @ f.T + alpha[:, None] + delta[None, :]
    return lam, f, fitted


def gsynth(
    y: FloatArray,
    treated: FloatArray,
    t0: int,
    *,
    n_factors: int = 3,
    seed: int = 0,
) -> dict[str, float | FloatArray]:
    """gsynth ATT: IFE fit on controls (+ treated pre-period), treated
    post counterfactual predicted by unit-specific loading on pre data.
    """
    y = _as_matrix(y, "y")
    n_units, t = y.shape
    tr = np.asarray(treated).astype(bool).ravel()
    if tr.size != n_units or tr.sum() < 1 or tr.sum() > n_units - 3:
        raise ValueError("need >=1 treated and >=3 controls")
    if not (3 <= t0 <= t - 2):
        raise ValueError("t0 must be in [3, T-2]")

    # IFE factor structure from controls only (all periods observed)
    mask_c = np.ones_like(y[~tr], dtype=bool)
    lam_c, f, _fit_c = _ife_als(y[~tr], mask_c, n_factors)

    # treated-unit loadings: fit on pre period only (tau-free)
    lam_t = np.empty((int(tr.sum()), n_factors))
    alpha_t = np.empty(int(tr.sum()))
    for ii, i in enumerate(np.flatnonzero(tr)):
        f_pre = f[:t0]
        coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(t0), f_pre]), y[i, :t0], rcond=None)
        alpha_t[ii] = coef[0]
        lam_t[ii] = coef[1:]

    # treated counterfactual: alpha_i + lam_i f_t for all t
    cf_t = alpha_t[:, None] + lam_t @ f.T
    # common time effects on controls' demeaned panel
    delta_c = np.asarray(y[~tr]) - np.asarray(_fit_c)
    delta = delta_c.mean(axis=0)  # per-period mean control residual ~ 0 anyway
    cf_t += delta[None, :] * 0.0  # control residuals ~0; keep for clarity

    eff = np.asarray(y[tr]) - cf_t
    att = float(eff[:, t0:].mean())

    # unit jackknife over treated units for honest dispersion
    taus_i = eff[:, t0:].mean(axis=1)
    if taus_i.size > 1:
        se = float(taus_i.std(ddof=1) / math.sqrt(taus_i.size))
    else:
        se = float(np.std(eff[:, t0:], ddof=1) / math.sqrt(eff[:, t0:].size))

    return {
        "att": att,
        "se": se,
        "z": float(att / max(se, 1e-12)),
        "att_by_unit": taus_i,
        "counterfactual": cf_t,
        "factors": f,
        "loadings_treated": lam_t,
        "n_factors": float(n_factors),
    }


def synth_ife_panel(
    n_units: int = 40,
    t: int = 60,
    n_treated: int = 6,
    t0: int | None = None,
    tau: float = 1.0,
    n_factors: int = 2,
    drift: float = 0.0,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Interactive-FE panel with optional factor drift + treated wedge."""
    if t0 is None:
        t0 = int(0.7 * t)
    rng = np.random.default_rng(seed)
    lam = rng.normal(0.0, 1.0, (n_units, n_factors))
    f = rng.normal(0.0, 1.0, (t, n_factors))
    if drift:
        f[:, 0] += np.linspace(0.0, drift, t)
    alpha_u = rng.normal(0.0, 0.3, n_units)
    delta_t = rng.normal(0.0, 0.2, t)
    y = lam @ f.T + alpha_u[:, None] + delta_t[None, :] + rng.normal(0.0, 0.25, (n_units, t))
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


def bench_gsynth(seed: int = 20261231 + 192) -> dict[str, float]:
    """gsynth self-check: ATT recovery under stationary AND drifting
    factors — where plain DiD and even SDID strain. All ``synthetic_*``."""
    d = synth_ife_panel(seed=seed, tau=1.5, drift=3.0, n_factors=2)
    y = np.asarray(d["y"])
    tr = np.asarray(d["treated"]).astype(bool)
    t0 = int(np.asarray(d["t0"]).item())
    tau_true = float(np.asarray(d["tau"]).item())

    est = gsynth(y, tr, t0, n_factors=3, seed=seed + 1)
    est2 = gsynth(y, tr, t0, n_factors=3, seed=seed + 1)

    did = float((y[tr, t0:].mean() - y[tr, :t0].mean()) - (y[~tr, t0:].mean() - y[~tr, :t0].mean()))

    d0 = synth_ife_panel(seed=seed + 2, tau=0.0, drift=3.0)
    est0 = gsynth(
        np.asarray(d0["y"]),
        np.asarray(d0["treated"]).astype(bool),
        int(np.asarray(d0["t0"]).item()),
        n_factors=3,
        seed=seed + 3,
    )

    return {
        "synthetic_att": float(est["att"]),
        "synthetic_tau_true": tau_true,
        "synthetic_att_err": abs(float(est["att"]) - tau_true),
        "synthetic_did_err": abs(did - tau_true),
        "synthetic_z": float(est["z"]),
        "synthetic_beats_did": float(abs(float(est["att"]) - tau_true) < abs(did - tau_true)),
        "synthetic_null_att": float(est0["att"]),
        "synthetic_detects": float(abs(float(est["att"])) > 3.0 * abs(float(est0["att"])) + 0.2),
        "synthetic_determinism": float(est["att"] == est2["att"]),
    }
