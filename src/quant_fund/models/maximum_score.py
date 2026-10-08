"""Manski's maximum score estimator for binary response (SYNTHETIC).

P(y=1|x) is only assumed to be non-decreasing in xβ (conditional
median zero on the disturbance) — no logit/probit link. The
estimator maximises the number of correctly predicted signs:

  S(b) = mean[ 1(y=1)·1(xb≥0) + 1(y=0)·1(xb<0) ]

S(b) is discontinuous, so the optimum is located by the Horowitz
smoothed score (a kernel mollifier on xb) plus a grid-over-Latin-
hypercube global search — the empirical-maximisation algorithm
that dominates naive grid search for k ≤ 4.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure β-direction recovery on
generated binary response — never market evidence.

References:
- Manski, C. F. (1975). Maximum score estimation of the
  stochastic utility model of choice. *J. Econometrics* 3,
  205-228.
- Manski, C. F. (1985). Semiparametric analysis of discrete
  response. *J. Econometrics* 27, 313-333.
- Horowitz, J. L. (1992). A smoothed maximum score estimator
  for the binary response model. *Econometrica* 60, 505-531 —
  the kernel-smoothed criterion used here.
- Horowitz, J. L. (1993). Semiparametric estimation of a
  work-trip mode choice model. *J. Econometrics* 58 — the SMS
  rate and bandwidth rule (h = σ·n^{-1/5}).

Composition: pure numpy — latin-hypercube global start, smoothed
score BFGS refine; deterministic ``np.random.default_rng``;
no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import norm

FloatArray = NDArray[np.float64]


def _as_xy(y: FloatArray, x: FloatArray) -> tuple[FloatArray, FloatArray]:
    yy = np.asarray(y, dtype=np.float64).ravel()
    xx = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if xx.shape[0] != yy.size:
        xx = xx.T
    if yy.size < 50 or xx.shape[0] != yy.size:
        raise ValueError("x row count must equal len(y), n>=50")
    if not np.all(np.isfinite(yy)) or not np.all(np.isfinite(xx)):
        raise ValueError("finite y and x required")
    u = np.unique(yy)
    if not np.array_equal(u, np.array([0.0, 1.0])):
        raise ValueError("y must be binary {0, 1} with both classes")
    if u.size != 2 or yy.mean() < 0.02 or yy.mean() > 0.98:
        raise ValueError("both outcome classes must have >=2% mass")
    if xx.shape[1] > 4:
        raise ValueError("maximum score supports <=4 regressors")
    if np.linalg.matrix_rank(xx) < xx.shape[1]:
        raise ValueError("regressors must have full column rank")
    return yy, xx


def _smooth_score(b: FloatArray, y: FloatArray, x: FloatArray, h: float) -> float:
    xb = x @ b
    # Horowitz kernel: 2Φ(xb/h) - 1 as smooth sign(xb)
    s = 2.0 * norm.cdf(xb / max(h, 1e-9)) - 1.0
    return float(np.mean((2.0 * y - 1.0) * s))


def _lhs(dim: int, n_pts: int, rng: np.random.Generator) -> FloatArray:
    """Latin-hypercube sample on the unit (dim-1)-sphere."""
    u = np.empty((n_pts, dim))
    for j in range(dim):
        u[:, j] = (rng.permutation(n_pts) + rng.uniform(size=n_pts)) / n_pts
    z = np.asarray(norm.ppf(np.clip(u, 1e-6, 1 - 1e-6)), dtype=np.float64)
    z = z / np.sqrt(np.sum(z * z, axis=1, keepdims=True))
    return z


def maximum_score(
    y: FloatArray,
    x: FloatArray,
    n_starts: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """Smoothed maximum score; β identified up to scale (||β||=1).

    Constant absorbed by normalising: first regressor's coefficient
    fixed by construction (scale normalisation β̂/||β̂||)."""
    yy, xx = _as_xy(y, x)
    n = yy.size
    k = xx.shape[1]
    # include intercept inside index; standardise regressors for scale
    mu, sd = xx.mean(0), xx.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    xs = (xx - mu) / sd
    x1 = np.column_stack([np.ones(n), xs])
    h = 1.06 * float(np.std(x1 @ np.zeros(k + 1) + 1.0)) * n ** (-1 / 5)
    h = max(h, 0.15)

    rng = np.random.default_rng(seed)
    starts = _lhs(k + 1, n_starts, rng)
    raw = np.array([float(np.mean((2 * yy - 1) * np.sign(x1 @ b))) for b in starts])
    order = np.argsort(-raw)[:8]
    best_b, best_s = starts[order[0]], -np.inf
    for idx in order:
        res = minimize(
            lambda b: -_smooth_score(b, yy, x1, h),
            starts[idx],
            method="Nelder-Mead",
            options={"maxiter": 400, "xatol": 1e-4},
        )
        b = res.x
        nb = np.linalg.norm(b)
        if nb < 1e-9:
            continue
        b = b / nb
        s = float(np.mean((2 * yy - 1) * np.sign(x1 @ b)))
        if s > best_s:
            best_s, best_b = s, b

    if best_s == -np.inf:
        raise ValueError("maximum score found no feasible direction")
    score_rate = float(np.mean(((x1 @ best_b) >= 0) == (yy > 0.5)))

    # reference: logistic fit direction
    try:
        eta = np.clip(np.log(np.clip(yy, 0.05, 0.95) / np.clip(1 - yy, 0.05, 0.95)), -8, 8)
        b_lr = np.linalg.lstsq(x1, eta, rcond=None)[0]
        b_lr = b_lr / np.linalg.norm(b_lr)
    except np.linalg.LinAlgError:
        b_lr = np.zeros(k + 1)
    # sign alignment
    for cand in (best_b, -best_b):
        if float(cand @ b_lr) >= 0:
            best_b = cand
            break

    return {
        "n": float(n),
        "score_rate": score_rate,
        "smoothed_h": h,
        **{f"beta_{j}": float(v) for j, v in enumerate(best_b)},
    }


def synth_binary_response(
    n: int = 1200,
    beta: tuple[float, ...] = (0.8, -0.5),
    hetero: bool = False,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Binary-response DGP under a heavy-tailed index model —
    y = 1{xβ + ε > 0}, ε Student-t (heteroskedastic optional) —
    a design where logit/probit links are misspecified."""
    rng = np.random.default_rng(seed)
    k = len(beta)
    x = np.column_stack([rng.normal(0.0, 1.0, n) for _ in range(k)])
    idx = x @ np.array(beta)
    eps = rng.standard_t(df=3, size=n)
    if hetero:
        eps = eps * (0.4 + 0.8 * np.abs(x[:, 0]))
    y = (idx + eps > 0).astype(np.float64)
    return {"y": y, "x": x, "beta_true": np.array(beta)}


def bench_maximum_score(seed: int = 20261231 + 221) -> dict[str, float]:
    """Maximum-score self-check: direction recovered under Student-t
    errors + heteroskedasticity — where a fixed link is misspecified.
    All ``synthetic_*``."""
    beta = (0.8, -0.5)
    d = synth_binary_response(beta=beta, hetero=True, seed=seed)
    out = maximum_score(np.asarray(d["y"]), np.asarray(d["x"]), seed=seed)
    b_true = np.array(beta) / np.linalg.norm(np.array(beta))
    b_hat = np.array([float(out["beta_1"]), float(out["beta_2"])])
    b_hat = b_hat / np.linalg.norm(b_hat)
    # sign convention: align with truth
    if float(b_hat @ b_true) < 0:
        b_hat = -b_hat
    corr = float(b_hat @ b_true)
    d0 = synth_binary_response(beta=(0.0, 0.0), seed=seed + 1)
    out0 = maximum_score(np.asarray(d0["y"]), np.asarray(d0["x"]), seed=seed)
    out_b = maximum_score(np.asarray(d["y"]), np.asarray(d["x"]), seed=seed)

    return {
        "synthetic_direction_cos": corr,
        "synthetic_score_rate": float(out["score_rate"]),
        "synthetic_beta1": float(out["beta_1"]),
        "synthetic_beta2": float(out["beta_2"]),
        "synthetic_null_score": float(out0["score_rate"]),
        "synthetic_detects": float(corr > 0.9 and out["score_rate"] > 0.62),
        "synthetic_determinism": float(float(out["beta_1"]) == float(out_b["beta_1"])),
    }
