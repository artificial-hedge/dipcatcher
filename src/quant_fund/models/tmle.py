"""Targeted maximum likelihood estimation (TMLE) of the ATE.

References
----------
- van der Laan, M.J. & Rubin, D. (2006). "Targeted Maximum Likelihood
  Learning." *The International Journal of Biostatistics* 2(1).
- van der Laan, M.J. & Rose, S. (2011). *Targeted Learning*. Springer.
- Gruber, S. & van der Laan, M. (2009). "Targeted Maximum Likelihood
  Estimation: A Gentle Introduction." U.C. Berkeley Division of
  Biostatistics Working Paper 252.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence.

Composition notes
-----------------
TMLE complements ``aipw_ate`` (wave 47): both are doubly robust under
misspecification of either the outcome or the propensity model, but TMLE
substitutes a one-dimensional *fluctuation submodel*

    logit Q*(A, X) = logit Q0(A, X) + eps * H(A, X),
    H(A, X) = A / g(X) - (1 - A) / (1 - g(X))

for the re-weighted estimator. Because Q* is a fitted probability it
respects the parameter space (unlike IPW, which can leave the support),
and the influence curve

    IC = H(A, X) * (Y - Q*(A, X)) + Q*(1, X) - Q*(0, X) - psi

gives a consistent standard error under either consistent Q0 or g.
The synthetic binary-outcome generator uses a nonlinear propensity that a
main-effects logit misspecifies; TMLE still recovers the risk difference.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _design(x: FloatArray) -> FloatArray:
    n = x.shape[0]
    return np.column_stack([np.ones(n), x])


def _irls_logit(y: FloatArray, x: FloatArray, iters: int = 50) -> FloatArray:
    xx = _design(x)
    beta = np.zeros(xx.shape[1])
    lam = 1e-6
    for _ in range(iters):
        z = xx @ beta
        p = 1.0 / (1.0 + np.exp(-z))
        w = np.maximum(p * (1.0 - p), 1e-9)
        h = xx.T @ (xx * w[:, None]) + lam * np.eye(xx.shape[1])
        g = xx.T @ (y - p) - lam * beta
        step = np.linalg.solve(h, g)
        beta = beta + step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return beta


def _ridge_fit(y: FloatArray, x: FloatArray, lam: float = 1e-6) -> FloatArray:
    xx = _design(x)
    return np.linalg.solve(xx.T @ xx + lam * np.eye(xx.shape[1]), xx.T @ y)


def _logit(z: FloatArray) -> FloatArray:
    return 1.0 / (1.0 + np.exp(-z))


def tmle_ate(
    y: FloatArray,
    treat: FloatArray,
    x: FloatArray,
    n_target_steps: int = 3,
    trim: float = 0.01,
) -> dict[str, float]:
    """TMLE estimate of E[Y(1) - Y(0)] for binary outcome ``y``.

    Initial outcome fit is a ridge-logit on (treat, x); the propensity is
    an IRLS logit on x. ``n_target_steps`` fluctuation updates are applied
    (converges in one step to numerical precision).
    """
    yy = np.asarray(y, dtype=np.float64)
    tt = np.asarray(treat, dtype=np.float64)
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim == 1:
        xx = xx[:, None]
    if yy.ndim != 1 or tt.ndim != 1 or yy.shape[0] != tt.shape[0] != xx.shape[0]:
        raise ValueError("y, treat, x must share n rows")
    n = yy.shape[0]
    if n < 60:
        raise ValueError("need n >= 60")
    if not np.all(np.isfinite(yy)) or not np.all(np.isfinite(tt)) or not np.all(np.isfinite(xx)):
        raise ValueError("non-finite inputs")
    if not np.all((tt == 0.0) | (tt == 1.0)):
        raise ValueError("treat must be 0/1")
    if not np.all((yy >= 0.0) & (yy <= 1.0)):
        raise ValueError("binary-outcome TMLE requires 0 <= y <= 1")

    gb = _irls_logit(tt, xx)
    g = np.clip(_logit(_design(xx) @ gb), trim, 1.0 - trim)

    xa = np.column_stack([tt, xx])
    qb = _irls_logit(yy, xa)
    x1 = np.column_stack([np.ones(n), xx])
    x0 = np.column_stack([np.zeros(n), xx])

    q1 = _logit(_design(x1) @ qb)
    q0 = _logit(_design(x0) @ qb)
    qa = np.where(tt == 1.0, q1, q0)

    eps_final = 0.0
    for _ in range(n_target_steps):
        h = tt / g - (1.0 - tt) / (1.0 - g)
        s = np.log(np.clip(qa, 1e-9, 1.0 - 1e-9) / np.clip(1.0 - qa, 1e-9, 1.0))
        denom = float(h @ h)
        if denom <= 0:
            raise ValueError("degenerate fluctuation")
        eps = float(h @ (yy - qa) / denom)
        qa = _logit(s + eps * h)
        q1 = _logit(np.log(np.clip(q1, 1e-9, 1.0 - 1e-9) / np.clip(1.0 - q1, 1e-9, 1.0)) + eps / g)
        q0 = _logit(
            np.log(np.clip(q0, 1e-9, 1.0 - 1e-9) / np.clip(1.0 - q0, 1e-9, 1.0)) - eps / (1.0 - g)
        )
        eps_final = eps

    psi = float(np.mean(q1 - q0))
    ic = (tt / g - (1.0 - tt) / (1.0 - g)) * (yy - qa) + (q1 - q0) - psi
    se = float(np.std(ic, ddof=1) / np.sqrt(n))

    q0r = _ridge_fit(yy[tt == 0], xx[tt == 0])
    q1r = _ridge_fit(yy[tt == 1], xx[tt == 1])
    m1 = _design(xx) @ q1r
    m0 = _design(xx) @ q0r
    ipw_aipw = float(np.mean(m1 - m0 + tt * (yy - m1) / g - (1.0 - tt) * (yy - m0) / (1.0 - g)))

    return {
        "ate_tmle": psi,
        "se": se,
        "ate_aipw": ipw_aipw,
        "epsilon": eps_final,
        "e_min": float(np.min(np.minimum(g, 1.0 - g))),
        "naive_diff": float(np.mean(yy[tt == 1]) - np.mean(yy[tt == 0])),
    }


def synth_tmle(
    n: int = 800,
    seed: int = 20261231 + 275,
    effect: float = 0.20,
) -> dict[str, FloatArray]:
    """Binary-outcome synth: nonlinear propensity + linear logit outcome.

    Propensity eta = -0.5 + 1.2*x1 + 0.8*x1*x2 - x2^2 (misspecified for a
    main-effects logit). Outcome logit = -0.4 + effect * a + 0.9*x1 - 0.6*x2
    (fit correctly), so TMLE tracks the true risk difference while the
    naive difference is confounded.
    """
    rng = np.random.default_rng(seed)
    if n < 60:
        raise ValueError("n too small")
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    eta = -0.5 + 1.2 * x1 + 0.8 * x1 * x2 - x2**2
    g = _logit(eta)
    a = (rng.uniform(0.0, 1.0, n) < g).astype(np.float64)
    lo = -0.4 + effect * a + 0.9 * x1 - 0.6 * x2
    y = (rng.uniform(0.0, 1.0, n) < _logit(lo)).astype(np.float64)
    return {
        "y": y,
        "treat": a,
        "x": np.column_stack([x1, x2]),
        "true_rd": np.full(
            n,
            _logit(-0.4 + effect + 0.9 * x1 - 0.6 * x2).mean()
            - _logit(-0.4 + 0.9 * x1 - 0.6 * x2).mean(),
        ),
    }


def bench_tmle(seed: int = 20261231 + 275) -> dict[str, float]:
    """Wave-48 self-check: TMLE recovers the risk difference under a
    misspecified propensity where the naive difference is confounded."""
    d = synth_tmle(seed=seed)
    rd = float(np.asarray(d["true_rd"])[0])
    a = tmle_ate(np.asarray(d["y"]), np.asarray(d["treat"]), np.asarray(d["x"]))
    b = tmle_ate(np.asarray(d["y"]), np.asarray(d["treat"]), np.asarray(d["x"]))
    detects = float(abs(a["ate_tmle"] - rd) < abs(a["naive_diff"] - rd) * 0.5)
    return {
        "synthetic_detects": detects,
        "synthetic_determinism": float(a == b),
        "synthetic_ate_tmle": a["ate_tmle"],
        "synthetic_true_rd": rd,
        "synthetic_naive_diff": a["naive_diff"],
        "synthetic_ate_aipw": a["ate_aipw"],
        "synthetic_se": a["se"],
        "synthetic_e_min": a["e_min"],
        "synthetic_epsilon": a["epsilon"],
    }
