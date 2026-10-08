"""Corradi-Swanson out-of-sample predictive accuracy test.

References
----------
- Corradi, V. & Swanson, N.R. (2006). "Predictive Density
  Evaluation." In *Handbook of Economic Forecasting* vol. 1,
  197-284.
- Corradi, V. & Swanson, N.R. (2007). "Nonparametric Bootstrap
  Procedures for Predictive Accuracy Based on Recursive
  Estimation Schemes." *Econometric Reviews* 26(4), 449-479.
- Diebold, F.X. & Mariano, R. (1995). "Comparing Predictive
  Accuracy." *JBES* 13(3), 253-263.
- Clark, T. & McCracken, M. (2001). "Tests of Equal Forecast
  Accuracy and Encompassing for Nested Models." *Journal of
  Econometrics* 105(1), 85-110.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence.

Composition notes
-----------------
Two nested models forecast ``y`` out of sample by a recursive
(expanding-window) scheme; the test statistic is the out-of-sample
MSPE difference ``f = mean(e0^2 - e1^2)`` over the P forecast
origin points. For nested models the Clark-McCracken / Corradi-
Swanson refinement applies: the null variance is degenerate when
the bigger model nests the smaller (both converge to the same
forecast under H0), so the raw Diebold-Mariano normal is
anti-conservative — we use the recursive moving-block bootstrap
(Corradi-Swanson 2007): resample the forecast-error differential
``d_t = e0_t^2 - e1_t^2`` in blocks of expected length ``b``, keep
``P`` replicates of the centered differential mean, and take the
bootstrap distribution for the p-value. ``enc_new`` reports the
Clark-McCracken ENC-NEW statistic ``P * mean(c_t)`` where
``c_t = e0_t^2 - e0_t*e1_t`` for completeness. The synth plants
an AR(1) DGP where the small model omits the true lag (small
miss) and where it nests correctly (null case), and the bench
requires rejection under the alternative and non-rejection under
the null at nominal 10%.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _ols_forecast(y: FloatArray, x_lags: int, r_end: int, n_oos: int) -> FloatArray:
    """Recursive expanding-window OLS one-step forecasts of an AR(x_lags).

    Returns the P=n_oos forecast errors for origins t = r_end-1 .. n-2.
    """
    errs = np.empty(n_oos)
    for i in range(n_oos):
        t = r_end - 1 + i  # forecast y_{t+1} using data ..t
        tr = y[: t + 1]
        if tr.size <= x_lags + 2:
            raise ValueError("insufficient training sample")
        rows = np.column_stack(
            [np.ones(tr.size - x_lags)]
            + [tr[x_lags - k - 1 : tr.size - k - 1] for k in range(x_lags)]
        )
        coef, *_ = np.linalg.lstsq(rows, tr[x_lags:], rcond=None)
        xr = np.concatenate([[1.0], tr[t - np.arange(x_lags)]])
        errs[i] = y[t + 1] - float(xr @ coef)
    return errs


def _mbb_pvalue(d: FloatArray, stat: float, b: int, n_boot: int, seed: int) -> tuple[float, float]:
    """Moving-block bootstrap p-value for mean(d) under the null."""
    p = d.size
    rng = np.random.default_rng(seed)
    dc = d - d.mean()
    reps = np.empty(n_boot)
    n_blocks = int(np.ceil(p / b))
    for r_i in range(n_boot):
        idx = np.empty(n_blocks * b, dtype=np.int64)
        starts = rng.integers(0, p - b + 1, size=n_blocks)
        for j, s in enumerate(starts):
            idx[j * b : (j + 1) * b] = np.arange(s, s + b)
        reps[r_i] = float(np.mean(dc[idx[:p]]))
    boot_stat = reps + 0.0  # centered already
    # one-sided: under alternative mean(d) > 0 (big model better)
    pval = float(np.mean(boot_stat >= stat))
    return pval, float(np.percentile(boot_stat, 95.0))


def cs_test(
    y: FloatArray,
    lags_small: int = 0,
    lags_big: int = 2,
    split: float = 0.5,
    block: int = 8,
    n_boot: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    """Corradi-Swanson OOS predictive accuracy for nested AR models.

    ``lags_small``=0 forecasts with intercept only (white-noise
    model); ``lags_big`` is the nesting AR order. Returns
    ``mspe_diff`` (small minus big, positive => big model better),
    ``f_stat`` (P-scaled differential), ``p_boot`` (moving-block
    bootstrap p-value), ``enc_new`` (Clark-McCracken ENC-NEW) and
    ``crit_95`` (bootstrap 95% critical value).
    """
    yy = np.asarray(y, dtype=np.float64)
    if yy.ndim != 1 or yy.size < 120 or not np.all(np.isfinite(yy)):
        raise ValueError("bad series")
    n = yy.size
    r_end = int(n * split)
    n_oos = n - r_end
    e0 = _ols_forecast(yy, lags_small, r_end, n_oos)
    e1 = _ols_forecast(yy, lags_big, r_end, n_oos)
    d = e0 * e0 - e1 * e1
    mspe_diff = float(np.mean(d))
    f_stat = float(np.mean(d) / (np.std(d, ddof=1) / np.sqrt(n_oos)))
    p_boot, crit = _mbb_pvalue(d, float(np.mean(d)), block, n_boot, seed)
    c = e0 * e0 - e0 * e1
    enc_new = float(n_oos * np.mean(c))
    return {
        "mspe_diff": mspe_diff,
        "f_stat": f_stat,
        "p_boot": p_boot,
        "crit_95": crit,
        "enc_new": enc_new,
        "mspe_small": float(np.mean(e0 * e0)),
        "mspe_big": float(np.mean(e1 * e1)),
        "p": float(n_oos),
    }


def synth_cs(
    seed: int = 20261231 + 305,
    t: int = 900,
    phi: float = 0.7,
    under_null: bool = False,
) -> FloatArray:
    """SYNTHETIC AR(2) (alternative) or AR(0) (null) series."""
    rng = np.random.default_rng(seed)
    x = np.empty(t)
    x[:2] = 0.0
    eps = rng.standard_normal(t)
    for i in range(2, t):
        x[i] = 0.0 if under_null else phi * x[i - 1] - 0.25 * phi * x[i - 2]
        x[i] += eps[i]
    return x


def bench_corradi_swanson(seed: int = 20261231 + 305) -> dict[str, float]:
    """Wave-53 self-check: reject under alternative, don't under null."""
    y_alt = synth_cs(seed=seed)
    r_alt = cs_test(y_alt, lags_small=0, lags_big=2, n_boot=300, seed=seed)
    y_null = synth_cs(seed=seed + 1, under_null=True)
    r_null = cs_test(y_null, lags_small=0, lags_big=2, n_boot=300, seed=seed)
    ok = r_alt["mspe_diff"] > 0.0 and r_alt["p_boot"] < 0.1 and r_null["p_boot"] > 0.1
    return {
        "synthetic_mspe_diff_alt": r_alt["mspe_diff"],
        "synthetic_p_alt": r_alt["p_boot"],
        "synthetic_p_null": r_null["p_boot"],
        "synthetic_enc_new_alt": r_alt["enc_new"],
        "synthetic_score": float(ok),
    }
