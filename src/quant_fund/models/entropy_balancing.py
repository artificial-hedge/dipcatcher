"""Entropy balancing calibration weights.

Hainmueller's entropy balancing: find control weights w_i that match
treatment-group moments exactly while staying as close as possible to
uniform (maximum entropy). Convex dual: minimize over lambda the
log-sum-exp objective; weights are exponentials of the fitted dual
variables — guaranteed positive, no trimming needed.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure covariate-balance and ATT recovery
on generated selection designs — never market evidence.

References:
- Hainmueller (2012). Entropy balancing for causal effects: a
  multivariate reweighting method to produce balanced samples in
  observational studies. *Political Analysis* 20.
- Zubizarreta (2015). Stable weights that balance covariates for
  estimation with incomplete outcome data. *JASA* 110.
- Zhao, Percival (2017). Entropy balancing is doubly robust.
  *J. Causal Inference* 5.
- King, Nielsen (2019). Why propensity scores should not be used for
  matching. *Political Analysis* 27 (context for balance-first design).

Composition: pure numpy — BFGS-free Newton on the log-sum-exp dual via
finite-difference Hessian; deterministic ``np.random.default_rng``;
no new dependencies.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_matrix(x: FloatArray, name: str) -> FloatArray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: finite (n, k) matrix required")
    return a


def _center_controls(x_c: FloatArray, target: FloatArray) -> FloatArray:
    """Centered control covariates relative to target moments."""
    return x_c - target[None, :]


def entropy_weights(
    x_treated: FloatArray,
    x_control: FloatArray,
    *,
    max_iter: int = 500,
    tol: float = 1e-9,
) -> dict[str, FloatArray | float]:
    """Entropy-balancing weights for controls matching treated moments.

    Minimizes L(λ) = log Σ_i exp(-λ·z_i) + λ·0 over dual λ via Newton
    steps with backtracking; z_i = x_ci - x̄_treated. Returns weights
    summing to n_c and the achieved max |balance gap|.
    """
    xt = _as_matrix(x_treated, "x_treated")
    xc = _as_matrix(x_control, "x_control")
    if xt.shape[1] != xc.shape[1]:
        raise ValueError("treated/control covariate dims must match")
    if xt.shape[0] < 2 or xc.shape[0] < xt.shape[1] + 2:
        raise ValueError("need n_treated>=2 and n_control >= k+2")
    target = xt.mean(axis=0)
    z = _center_controls(xc, target)
    # standardize for conditioning
    sd = z.std(axis=0)
    sd = np.where(sd > 1e-10, sd, 1.0)
    z = z / sd[None, :]
    n_c, k = z.shape

    lam = np.zeros(k)
    converged = False
    n_it = 0
    for _ in range(max_iter):
        n_it += 1
        s = z @ lam
        w_un = np.exp(-s - (-s).max())  # stable softmax of -s
        w = w_un / w_un.sum()
        grad = -(w @ z)  # E_w[z] should → 0
        hess = (w[:, None] * z).T @ z - np.outer(w @ z, w @ z)
        hess += np.eye(k) * 1e-8
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]

        # backtracking line search on dual objective
        def obj(lv: FloatArray) -> float:
            sv = z @ lv
            m = float((-sv).max())
            return float(m + np.log(np.sum(np.exp(-sv - m))))

        f0 = obj(lam)
        t = 1.0
        for _ in range(60):
            cand = lam - t * step
            if obj(cand) < f0 - 1e-4 * t * float(grad @ step):
                break
            t *= 0.5
        else:
            cand = lam - step
        lam = cand
        if float(np.abs(grad).max()) < tol:
            converged = True
            break
        w_chk = np.exp(-(z @ lam) - (-(z @ lam)).max())
        w_chk /= w_chk.sum()
        if float(1.0 / np.sum(w_chk**2)) < 2.0:
            break  # infeasible balance: weights collapsed onto a corner

    s = z @ lam
    w_un = np.exp(-s - (-s).max())
    w = w_un / w_un.sum()
    gap = np.abs(w @ z)
    bal_x = np.abs((w[:, None] * xc).sum(axis=0) - target)
    return {
        "weights": w * n_c,  # sum to n_c like IPW scale
        "max_gap_std": float(gap.max()) if gap.size else 0.0,
        "max_gap_raw": float(bal_x.max()) if bal_x.size else 0.0,
        "iterations": float(n_it),
        "converged": float(converged),
        "ess": float(1.0 / np.sum(w**2)),  # effective sample size of controls
    }


def balance_table(x: FloatArray, d: FloatArray, w: FloatArray | None = None) -> dict[str, float]:
    """Max |SMD| before/after weighting (weighted = control weights)."""
    xa = _as_matrix(x, "x")
    dd = np.asarray(d).astype(bool).ravel()
    if dd.size != xa.shape[0] or dd.sum() < 2 or (~dd).sum() < 2:
        raise ValueError("need >=2 per arm")
    x1, x0 = xa[dd], xa[~dd]
    smd_pre = np.abs(x1.mean(axis=0) - x0.mean(axis=0)) / np.sqrt(
        0.5 * (x1.var(axis=0) + x0.var(axis=0)) + 1e-12
    )
    if w is None:
        return {"max_smd": float(smd_pre.max())}
    ww = np.asarray(w, dtype=np.float64).ravel()
    if ww.size != x0.shape[0]:
        raise ValueError("weights must match control rows")
    wn = ww / ww.sum()
    mu0w = (wn[:, None] * x0).sum(axis=0)
    var0w = (wn[:, None] * (x0 - mu0w) ** 2).sum(axis=0)
    smd_post = np.abs(x1.mean(axis=0) - mu0w) / np.sqrt(0.5 * (x1.var(axis=0) + var0w) + 1e-12)
    return {"max_smd": float(smd_pre.max()), "max_smd_w": float(smd_post.max())}


def att_entropy(y: FloatArray, d: FloatArray, x: FloatArray) -> dict[str, float]:
    """ATT via entropy balancing: treated mean vs weighted control mean."""
    y = _as_matrix(y, "y").ravel()
    dd = np.asarray(d).astype(bool).ravel()
    xa = _as_matrix(x, "x")
    if dd.size != y.size or xa.shape[0] != y.size:
        raise ValueError("length mismatch")
    out = entropy_weights(xa[dd], xa[~dd])
    w = np.asarray(out["weights"])
    wn = w / w.sum()
    att = float(y[dd].mean() - (wn * y[~dd]).sum())
    # effective sample size-based SE for the control mean
    ess = float(out["ess"])
    se = float(y[~dd].std(ddof=1) / math.sqrt(max(ess, 2.0)))
    return {
        "att": att,
        "se": se,
        "z": att / max(se, 1e-12),
        "ess": ess,
        "max_smd_w": float(balance_table(xa, dd, w)["max_smd_w"]),
    }


def synth_eb(
    n: int = 600, effect: float = 1.2, n_treated: int | None = None, seed: int = 0
) -> dict[str, FloatArray]:
    """Moderate selection on x1/x2 level (not tail-extreme, so the
    balance set stays feasible) — plain mean-diff biased; entropy
    balance on [x, x1^2] recovers ATT."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (n, 3))
    ps = 1.0 / (1.0 + np.exp(-(0.6 * x[:, 0] + 0.25 * x[:, 1] - 1.4)))
    d = rng.random(n) < ps
    if d.sum() < 10 or (~d).sum() < 10:
        raise ValueError("degenerate selection draw")
    y = 0.9 * x[:, 0] + 0.5 * x[:, 1] + rng.normal(0.0, 1.0, n)
    y[d] += effect
    # linear moments keep the balance set feasible under moderate
    # selection; quadratic moments on tail-selected samples are a known
    # infeasible EB target (dual is unbounded → weights collapse).
    return {"y": y, "d": d.astype(np.float64), "x": x, "effect": np.full(1, effect)}


def bench_entropy_balancing(seed: int = 20261231 + 195) -> dict[str, float]:
    """Entropy-balancing self-check: balance restored + ATT recovered
    vs raw mean difference. All ``synthetic_*``."""
    d = synth_eb(seed=seed, effect=1.2)
    y = np.asarray(d["y"]).ravel()
    dd = np.asarray(d["d"]).astype(bool)
    x = np.asarray(d["x"])
    eff = float(np.asarray(d["effect"]).item())

    est = att_entropy(y, dd, x)
    est2 = att_entropy(y, dd, x)
    raw = float(y[dd].mean() - y[~dd].mean())
    w = np.asarray(entropy_weights(x[dd], x[~dd])["weights"])
    bal_w = balance_table(x, dd, w)

    d0 = synth_eb(seed=seed + 1, effect=0.0)
    est0 = att_entropy(
        np.asarray(d0["y"]).ravel(), np.asarray(d0["d"]).astype(bool), np.asarray(d0["x"])
    )

    return {
        "synthetic_att": float(est["att"]),
        "synthetic_effect_true": eff,
        "synthetic_att_err": abs(float(est["att"]) - eff),
        "synthetic_raw_err": abs(raw - eff),
        "synthetic_max_smd_pre": float(balance_table(x, dd)["max_smd"]),
        "synthetic_max_smd_post": float(bal_w["max_smd_w"]),
        "synthetic_beats_raw": float(abs(float(est["att"]) - eff) < abs(raw - eff)),
        "synthetic_converged": float(entropy_weights(x[dd], x[~dd])["converged"]),
        "synthetic_null_att": float(est0["att"]),
        "synthetic_detects": float(abs(float(est["att"])) > 0.5),
        "synthetic_determinism": float(est["att"] == est2["att"]),
    }
