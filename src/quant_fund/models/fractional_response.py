"""Fractional response models — outcomes bounded in [0, 1] (SYNTHETIC).

For share/fraction outcomes (fraction allocated, hit-rate, utilisation),
the conditional mean is E[y|x] = G(xβ) with G a logit or probit CDF.
Papke-Wooldridge quasi-MLE is consistent for the conditional mean under
any within-unit error structure, unlike OLS which violates the bounds
and unlike log-odds transforms which drop boundary values (0s and 1s).

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure conditional-mean recovery on
generated fractional panels — never market evidence.

References:
- Papke, L. E., Wooldridge, J. M. (1996). Econometric methods for
  fractional response variables with an application to 401(k) plan
  participation rates. *J. Applied Econometrics* 11, 619-632.
- Papke, L. E., Wooldridge, J. M. (2008). Panel data methods for
  fractional response variables with an application to test pass
  rates. *J. Econometrics* 145, 121-133 — robust sandwich SEs.
- McCullagh, P., Nelder, J. A. (1989). *Generalized Linear Models*,
  2nd ed. — the Bernoulli quasi-likelihood used.
- Gourieroux, C., Monfort, A., Trognon, A. (1984). Pseudo maximum
  likelihood methods. *Econometrica* 52 — QMLE consistency.

Composition: pure numpy — Bernoulli quasi-likelihood MLE via IRLS,
robust sandwich SE, average marginal effects; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit
from scipy.stats import norm

FloatArray = NDArray[np.float64]


def _as_xy(y: FloatArray, x: FloatArray) -> tuple[FloatArray, FloatArray]:
    yy = np.asarray(y, dtype=np.float64).ravel()
    xx = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if xx.shape[0] != yy.size:
        xx = xx.T
    if yy.size < 30 or xx.shape[0] != yy.size:
        raise ValueError("x row count must equal len(y), n>=30")
    if not np.all(np.isfinite(yy)) or not np.all(np.isfinite(xx)):
        raise ValueError("finite y and x required")
    if np.any(yy < -1e-9) or np.any(yy > 1 + 1e-9):
        raise ValueError("fractional outcome must lie in [0, 1]")
    if np.unique(yy).size < 3:
        raise ValueError("outcome needs >=3 distinct values")
    return yy, xx


def fractional_fit(
    y: FloatArray,
    x: FloatArray,
    iters: int = 60,
) -> dict[str, float]:
    """Fractional logit quasi-MLE via IRLS.

    E[y|x] = Λ(xβ); score equations are the Bernoulli QMLE first-order
    conditions — valid for any [0,1]-valued y, including 0/1 mass."""
    yy, xx = _as_xy(y, x)
    n = yy.size
    xx1 = np.column_stack([np.ones(n), xx])
    b = np.linalg.lstsq(
        xx1, np.log(np.clip(yy, 0.01, 0.99) / np.clip(1 - yy, 0.01, 0.99)), rcond=None
    )[0]

    for _ in range(iters):
        p = expit(xx1 @ b)
        w = np.clip(p * (1 - p), 1e-8, None)
        # IRLS working response
        z = xx1 @ b + (yy - p) / w
        wx = xx1 * np.sqrt(w)[:, None]
        b_new = np.linalg.lstsq(wx, np.sqrt(w) * z, rcond=None)[0]
        if np.allclose(b_new, b, atol=1e-8):
            b = b_new
            break
        b = b_new
    p = expit(xx1 @ b)
    resid = yy - p
    w = np.clip(p * (1 - p), 1e-8, None)
    # quasi-likelihood sandwich: E[xx']/w bread, E[resid²xx']/w² meat
    bread = (xx1 * w[:, None]).T @ xx1 / n
    meat = (xx1 * (resid**2)[:, None]).T @ xx1 / n
    try:
        vcov = np.linalg.inv(bread) @ meat @ np.linalg.inv(bread) / n
    except np.linalg.LinAlgError:
        raise ValueError("singular information matrix") from None
    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))

    # OLS-on-y reference (ignores bounds)
    b_ols = np.linalg.lstsq(xx1, yy, rcond=None)[0]
    # average marginal effect of x_1
    ame = float(np.mean(p * (1 - p)) * b[1])
    # calibration: mean fitted vs mean y
    cal = float(abs(np.mean(p) - np.mean(yy)))
    # in-bounds prediction share
    pred_in = float(np.mean((p >= 0) & (p <= 1)))
    ols_pred = xx1 @ b_ols
    ols_in = float(np.mean((ols_pred >= 0) & (ols_pred <= 1)))

    return {
        "n": float(n),
        "ame_1": ame,
        "calibration": cal,
        "pred_inbounds": pred_in,
        "ols_pred_inbounds": ols_in,
        "mcFadden_r2": float(
            1.0
            - float(
                np.sum(
                    yy * np.log(np.clip(p, 1e-9, 1)) + (1 - yy) * np.log(np.clip(1 - p, 1e-9, 1))
                )
                / np.sum(
                    yy * np.log(np.clip(yy.mean(), 1e-9, 1))
                    + (1 - yy) * np.log(np.clip(1 - yy.mean(), 1e-9, 1))
                )
            )
        ),
        **{f"beta_{j}": float(v) for j, v in enumerate(b)},
        **{f"se_{j}": float(v) for j, v in enumerate(se)},
        "beta_ols_1": float(b_ols[1]),
        "z_1": float(b[1] / max(se[1], 1e-12)),
        "p_1": float(2 * (1 - norm.cdf(abs(float(b[1] / max(se[1], 1e-12)))))),
    }


def synth_fractional(
    n: int = 1200,
    beta: float = 0.8,
    hetero: bool = True,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Fractional DGP: y = clip(Λ(βx)+noise) — heavy mass at the
    boundaries so bounded predictions matter."""
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.normal(0.0, 1.0, n)])
    mu = expit(beta * x[:, 0])
    y = np.clip(mu + rng.normal(0.0, 0.06, n), 0.0, 1.0)
    return {"y": y, "x": x, "beta_true": np.array([beta])}


def bench_fractional_response(
    seed: int = 20261231 + 219,
) -> dict[str, float]:
    """Fractional-response self-check: QMLE recovers β, keeps every
    prediction in [0,1] (OLS breaches), calibrates the mean.
    All ``synthetic_*``."""
    d = synth_fractional(beta=1.4, seed=seed)
    out = fractional_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    d0 = synth_fractional(beta=0.0, seed=seed + 1)
    out0 = fractional_fit(np.asarray(d0["y"]), np.asarray(d0["x"]))
    out_b = fractional_fit(np.asarray(d["y"]), np.asarray(d["x"]))

    b1 = float(out["beta_1"])
    return {
        "synthetic_beta1": b1,
        "synthetic_beta1_err": float(abs(b1 - 1.4)),
        "synthetic_z1": float(out["z_1"]),
        "synthetic_p1": float(out["p_1"]),
        "synthetic_calibration": float(out["calibration"]),
        "synthetic_pred_inbounds": float(out["pred_inbounds"]),
        "synthetic_ols_inbounds": float(out["ols_pred_inbounds"]),
        "synthetic_null_beta1": float(abs(out0["beta_1"])),
        "synthetic_detects": float(abs(b1 - 1.4) < 0.25 and float(out["p_1"]) < 0.05),
        "synthetic_determinism": float(b1 == float(out_b["beta_1"])),
    }
