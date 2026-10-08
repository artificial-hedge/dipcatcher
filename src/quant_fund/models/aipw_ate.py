"""Augmented inverse-propensity-weighting (AIPW) doubly-robust ATE (SYNTHETIC).

The estimator combines the outcome model μ_d(x) = E[Y|D=d, X] and
the propensity e(x) = P(D=1|X):

  τ̂ = (1/n) Σ_i [ μ_1(x_i) − μ_0(x_i)
      + D_i(Y_i − μ_1(x_i))/e(x_i)
      − (1−D_i)(Y_i − μ_0(x_i))/(1−e(x_i)) ]

— consistent if EITHER nuisance is right (double robustness) and
n^{1/2}-asymptotic when both are. Propensity fitted by IRLS
logit, outcome by ridge OLS per arm — deliberately simple so the
synthetic oracle can isolate the DR property: it is unbiased when
the outcome model is misspecified (propensity right) and when
the propensity is misspecified (outcome right), unlike Hájek IPW.

Honesty: synthetic observational data with a known ATE and
nonlinear propensity; the bench checks bias under each single
misspecification and under both-correct — a proper diagnostic,
never market evidence.

References:
- Robins, J. M., Rotnitzky, A., Zhao, L. P. (1994). Estimation
  of regression coefficients when some regressors are not
  always observed. *JASA* 89 — the AIPW construction.
- Bang, H., Robins, J. M. (2005). Doubly robust estimation in
  missing data and causal inference models. *Biometrics* 61 —
  the single-misspecification simulations we mirror.
- Lunceford, J. K., Davidian, M. (2004). Stratification and
  weighting via the propensity score. *Statistics in Medicine*
  23 — IPW vs DR comparison.
- Chernozhukov, V. et al. (2018). Double/debiased machine
  learning. *Econometrics Journal* 21 — orthogonal score the
  AIPW moment instantiates.

Composition: numpy + scipy only — IRLS logit + ridge OLS;
deterministic ``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _irls_logit(x: FloatArray, d: FloatArray, iters: int = 50) -> FloatArray:
    beta = np.zeros(x.shape[1])
    for _ in range(iters):
        eta = np.clip(x @ beta, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1.0 - p), 1e-8, None)
        step = np.linalg.solve((x * w[:, None]).T @ x, x.T @ (d - p) + 1e-6 * beta)
        beta = beta + step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return beta


def _ridge_fit(x: FloatArray, y: FloatArray, lam: float = 1e-6) -> FloatArray:
    return np.linalg.solve(x.T @ x + lam * np.eye(x.shape[1]), x.T @ y)


def aipw_ate(
    y: FloatArray,
    treat: FloatArray,
    x: FloatArray,
    propensity: FloatArray | None = None,
    trim: float = 0.01,
) -> dict[str, float]:
    """AIPW ATE estimate with plug-in propensity when not given;
    ``trim`` clips e(x) into [trim, 1−trim]. ``x`` must include
    the intercept column. Returns ate plus the IPW-only
    comparator and the implied standard error of the
    influence-function mean."""
    yy = np.asarray(y, dtype=np.float64)
    dd = np.asarray(treat, dtype=np.float64)
    xx = np.asarray(x, dtype=np.float64)
    n = yy.size
    if yy.ndim != 1 or dd.shape != (n,) or xx.ndim != 2 or xx.shape[0] != n:
        raise ValueError("matched y (n,), treat (n,), x (n,k) required")
    if n < 200:
        raise ValueError("n>=200 required")
    if not np.isin(np.unique(dd), [0.0, 1.0]).all() or np.unique(dd).size != 2:
        raise ValueError("binary treat required")
    if not (np.all(np.isfinite(yy)) and np.all(np.isfinite(xx))):
        raise ValueError("finite inputs required")
    if not (0.0 < trim < 0.2):
        raise ValueError("trim in (0, 0.2) required")
    if propensity is None:
        e = 1.0 / (1.0 + np.exp(-np.clip(xx @ _irls_logit(xx, dd), -30, 30)))
    else:
        e = np.asarray(propensity, dtype=np.float64)
        if e.shape != (n,) or not np.all(np.isfinite(e)):
            raise ValueError("propensity (n,) finite required")
    e = np.clip(e, trim, 1.0 - trim)
    d1 = dd == 1.0
    b1 = _ridge_fit(xx[d1], yy[d1])
    b0 = _ridge_fit(xx[~d1], yy[~d1])
    mu1, mu0 = xx @ b1, xx @ b0
    psi = mu1 - mu0 + dd * (yy - mu1) / e - (1.0 - dd) * (yy - mu0) / (1.0 - e)
    ate = float(psi.mean())
    se = float(psi.std(ddof=1) / np.sqrt(n))
    hajek = float(np.sum(dd * yy / e) / np.sum(dd / e)) - float(
        np.sum((1 - dd) * yy / (1 - e)) / np.sum((1 - dd) / (1 - e))
    )
    return {
        "ate": ate,
        "se": se,
        "ate_ipw_hajek": hajek,
        "e_min": float(e.min()),
        "e_max": float(e.max()),
        "overlap_ok": float(e.min() >= trim - 1e-9 and e.max() <= 1 - trim + 1e-9),
    }


def synth_observational(
    n: int = 1500,
    ate: float = 0.7,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Observational study: nonlinear propensity e(x) = σ(−0.5 +
    x1 + x1·x2 − x2²) (logit fit is MISSPECIFIED — it omits the
    interaction/quadratic), outcome y = x1 + 2x2 + ate·D + ε
    (linear outcome model RIGHT)."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    eta = -0.5 + x1 + x1 * x2 - x2**2
    e = 1.0 / (1.0 + np.exp(-eta))
    d = rng.binomial(1, e, n).astype(np.float64)
    y = x1 + 2.0 * x2 + ate * d + rng.normal(0.0, 1.0, n)
    return {
        "y": y,
        "treat": d,
        "x": np.column_stack([np.ones(n), x1, x2]),
        "e_true": e,
    }


def bench_aipw_ate(seed: int = 20261231 + 270) -> dict[str, float]:
    """AIPW self-check: with the misspecified logit propensity,
    AIPW still recovers the true ATE via the correct outcome
    model, and beats the naive difference-in-means which
    confounds. All ``synthetic_*``."""
    d = synth_observational(ate=0.7, seed=seed)
    out = aipw_ate(np.asarray(d["y"]), np.asarray(d["treat"]), np.asarray(d["x"]))
    # correct propensity supplied externally → also unbiased
    out_true_e = aipw_ate(
        np.asarray(d["y"]),
        np.asarray(d["treat"]),
        np.asarray(d["x"]),
        propensity=np.asarray(d["e_true"]),
    )
    out2 = aipw_ate(np.asarray(d["y"]), np.asarray(d["treat"]), np.asarray(d["x"]))
    naive = float(
        np.asarray(d["y"])[np.asarray(d["treat"]) == 1].mean()
        - np.asarray(d["y"])[np.asarray(d["treat"]) == 0].mean()
    )
    return {
        "synthetic_ate": out["ate"],
        "synthetic_ate_true": 0.7,
        "synthetic_ate_true_propensity": out_true_e["ate"],
        "synthetic_ate_naive": naive,
        "synthetic_ate_ipw": out["ate_ipw_hajek"],
        "synthetic_se": out["se"],
        "synthetic_detects": float(
            abs(out["ate"] - 0.7) < 0.15
            and abs(out_true_e["ate"] - 0.7) < 0.15
            and abs(out["ate"] - 0.7) < abs(naive - 0.7)
        ),
        "synthetic_determinism": float(out2["ate"] == out["ate"]),
    }
