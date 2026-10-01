"""Censored quantile regression — Powell's CLAD estimator.

When the outcome is censored (e.g. top-coded at c), ordinary
quantile regression on the censored sample is biased. Powell's
censored least absolute deviations estimator drops observations
whose fitted value would breach the censoring boundary — the
censoring is itself informative only through the constraint.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure censored-median recovery on
generated top-coded panels — never market evidence.

References:
- Powell, J. L. (1986). Censored regression quantiles. *Journal of
  Econometrics* 32, 143-155 — the CLAD estimator.
- Powell, J. L. (1984). Least absolute deviations estimation for
  the censored regression model. *J. Econometrics* 25, 303-325.
- Chernozhukov, V., Hong, H. (2002). Three-step censored quantile
  regression and extramarital affairs. *JASA* 97, 872-882 —
  computational simplification used here (uncensored-proportion
  selection, then restricted QR).
- Fitzenberger, B. (1997). *Computational Aspects of Censored
  Quantile Regression* — algorithm notes.

Composition: pure numpy — Chernozhukov-Hong three-step (OLS censor
  estimate → keep pred < c → QR on kept subsample), median-τ
  focused; deterministic ``np.random.default_rng``; no new
  dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

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
    return yy, xx


def _quantile_reg(y: FloatArray, x: FloatArray, tau: float, iters: int = 400) -> FloatArray:
    """QR via iteratively-reweighted least squares on the check loss."""
    n, k = x.shape
    b = np.linalg.lstsq(x, y, rcond=None)[0]
    for _ in range(iters):
        r = y - x @ b
        w = 1.0 / np.maximum(np.abs(r), 0.02)  # |r| weighting ≈ L1
        if tau != 0.5:
            # tilt weights for general τ
            w = w * np.where(r >= 0, 2 * tau, 2 * (1 - tau))
        wxt = (x * w[:, None]).T
        b_new = np.linalg.solve(wxt @ x + 1e-8 * np.eye(k), wxt @ y)
        if np.allclose(b_new, b, atol=1e-8):
            b = b_new
            break
        b = b_new
    return np.asarray(b, dtype=np.float64)


def censored_quantile(
    y: FloatArray,
    x: FloatArray,
    censor_at: float,
    tau: float = 0.5,
) -> dict[str, float]:
    """Powell/CH censored-QR: estimate slope under top-coding at
    ``censor_at``. Three steps: (1) uncensored proportion via OLS,
    (2) drop units predicted above c, (3) QR on the kept sample."""
    yy, xx = _as_xy(y, x)
    n = yy.size
    if not 0.05 < tau < 0.95:
        raise ValueError("tau in (0.05, 0.95)")
    share_cens = float(np.mean(yy >= censor_at - 1e-12))
    if not 0.01 < share_cens < 0.9:
        raise ValueError("censoring share must be in (0.01, 0.9)")
    xx1 = np.column_stack([np.ones(n), xx])
    k = xx1.shape[1]

    # step 1-2: OLS propensity of being uncensored (CH simplification)
    b_ols = np.linalg.lstsq(xx1, yy, rcond=None)[0]
    keep = (xx1 @ b_ols) < censor_at
    if keep.sum() < k + 10:
        raise ValueError("too few uncensored observations")
    # step 3: QR on kept
    b_qr = _quantile_reg(yy[keep], xx1[keep], tau)
    # iterate the keep-set once (fixed-point polish)
    keep2 = (xx1 @ b_qr) < censor_at
    if keep2.sum() < k + 10:
        raise ValueError("iterated keep-set collapsed")
    b_qr = _quantile_reg(yy[keep2], xx1[keep2], tau)

    # naive QR ignoring censoring (biased reference)
    b_naive = _quantile_reg(yy, xx1, tau)

    resid = yy[keep2] - xx1[keep2] @ b_qr
    scale = float(np.median(np.abs(resid))) / 0.6744898
    return {
        "tau": tau,
        "censor_share": share_cens,
        "n_kept": float(keep2.sum()),
        "resid_scale": scale,
        **{f"beta_{j}": float(v) for j, v in enumerate(b_qr)},
        **{f"beta_naive_{j}": float(v) for j, v in enumerate(b_naive)},
    }


def synth_censored(
    n: int = 1200,
    beta: float = 1.0,
    censor_at: float = 1.5,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Top-coded DGP: y* = βx + ε, y = min(y*, c)."""
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.normal(0.0, 1.0, n)])
    ystar = beta * x[:, 0] + rng.normal(0.0, 0.6, n)
    y = np.minimum(ystar, censor_at)
    return {"y": y, "x": x, "beta_true": np.array([beta])}


def bench_censored_quantile(
    seed: int = 20261231 + 217,
) -> dict[str, float]:
    """Censored-QR self-check: corrected slope beats naive censored
    sample QR under heavy top-coding. All ``synthetic_*``."""
    d = synth_censored(beta=1.0, censor_at=1.5, seed=seed)
    out = censored_quantile(np.asarray(d["y"]), np.asarray(d["x"]), censor_at=1.5)
    d0 = synth_censored(beta=0.0, censor_at=0.4, seed=seed + 1)
    out0 = censored_quantile(np.asarray(d0["y"]), np.asarray(d0["x"]), censor_at=0.4)
    out_b = censored_quantile(np.asarray(d["y"]), np.asarray(d["x"]), censor_at=1.5)

    b1 = float(out["beta_1"])
    nb = float(out["beta_naive_1"])
    return {
        "synthetic_beta1": b1,
        "synthetic_beta1_err": float(abs(b1 - 1.0)),
        "synthetic_beta_naive": nb,
        "synthetic_naive_err": float(abs(nb - 1.0)),
        "synthetic_beats_naive": float(abs(b1 - 1.0) < abs(nb - 1.0)),
        "synthetic_censor_share": float(out["censor_share"]),
        "synthetic_n_kept": float(out["n_kept"]),
        "synthetic_null_beta1": float(abs(out0["beta_1"])),
        "synthetic_detects": float(abs(b1 - 1.0) < 0.25 and abs(float(out0["beta_1"])) < 0.3),
        "synthetic_determinism": float(b1 == float(out_b["beta_1"])),
    }
