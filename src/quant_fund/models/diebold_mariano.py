"""Diebold-Mariano (1995) + Harvey-Leybourne-Newbold (1997) tests.

References
----------
- Diebold, F.X. & Mariano, R.S. (1995). "Comparing Predictive
  Accuracy." *Journal of Business & Economic Statistics* 13(3),
  253-263.
- Harvey, D., Leybourne, S. & Newbold, P. (1997). "Testing the
  Equality of Prediction Mean Squared Errors."
  *International Journal of Forecasting* 13(2), 281-291.
- Diebold, F.X. (2015). "Comparing Predictive Accuracy, Twenty
  Years Later." *Journal of Business & Economic Statistics*
  33(1), 1-9.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
The DM statistic is a t-test on the loss-differential series
``d_t = L(e1_t) - L(e2_t)``: under the null of equal predictive
accuracy ``E[d_t] = 0``, and with a serially correlated
``d_t`` (h-step-ahead overlapping forecasts induce MA(h-1)) the
long-run variance must be estimated with HAC weights. We use
the canonical Bartlett long-run variance
``2*pi*f_d(0) = gamma_0 + 2*sum_{k<h}(1-k/h)*gamma_k`` and the
Harvey-Leybourne-Newbold small-sample correction, which scales
the DM statistic by ``sqrt((n+1-2h+h(h-1)/n)/n)`` and compares
against ``t_{n-1}`` rather than N(0,1). ``dm_test`` returns the
raw DM, the HLN-modified statistic, and a two-sided p on the
modified statistic. ``synth_dm`` plants two forecasts of an
AR(0.6) target where model 2 is exactly the conditional mean
(superior by construction) and model 1 adds idiosyncratic
noise — the bench gates on rejecting the equal-accuracy null
for the deliberately worse forecaster and *not* rejecting when
both errors are the same.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as _stats

FloatArray = NDArray[np.float64]


def dm_test(
    e1: FloatArray,
    e2: FloatArray,
    h: int = 1,
    loss: str = "se",
    power: float = 2.0,
) -> dict[str, float]:
    """Diebold-Mariano + Harvey-Leybourne-Newbold comparison."""
    a = np.asarray(e1, dtype=np.float64)
    b = np.asarray(e2, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size < 30:
        raise ValueError("bad errors")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("non-finite errors")
    if h < 1 or h > a.size // 4:
        raise ValueError("bad h")
    if loss == "se":
        d = a * a - b * b
    elif loss == "ae":
        d = np.abs(a) - np.abs(b)
    else:
        d = np.abs(a) ** power - np.abs(b) ** power
    n = d.size
    dbar = float(np.mean(d))
    # Bartlett long-run variance with h-1 nonzero autocovariances
    v = float(np.var(d, ddof=1))
    for k in range(1, h):
        g = float(np.cov(d[k:], d[:-k], ddof=1)[0, 1])
        v += 2.0 * (1.0 - k / h) * g
    if v <= 0:
        raise ValueError("degenerate loss differential")
    dm = dbar / np.sqrt(v / n)
    # HLN small-sample correction -> t_{n-1}
    adj = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = float(dm * adj)
    p = float(2.0 * _stats.t.sf(abs(dm_hln), n - 1))
    return {
        "dm": float(dm),
        "dm_hln": dm_hln,
        "p_hln": p,
        "mean_diff": dbar,
        "lrv": v,
    }


def synth_dm(
    seed: int = 20261231 + 317,
    n: int = 400,
    rho: float = 0.6,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """SYNTHETIC: AR target, model2=cond mean, model1=noisier."""
    rng = np.random.default_rng(seed)
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = rho * z[t - 1] + rng.normal()
    eps = np.diff(z)  # innovations of the AR (model-2 residual)
    e_good = eps.copy()
    e_bad = eps + rng.normal(0.0, 0.6, eps.size)
    # reversed copy: identical loss distribution, E[d]=0
    e_same = eps[::-1].copy()
    return np.asarray(e_bad), np.asarray(e_good), np.asarray(e_same)


def bench_diebold_mariano(
    seed: int = 20261231 + 317,
) -> dict[str, float]:
    """Wave-55 self-check: reject equal-accuracy when model1 worse."""
    e_bad, e_good, e_same = synth_dm(seed=seed)
    r_diff = dm_test(e_bad, e_good, h=1)
    r_same = dm_test(e_good, e_same, h=1)
    ok = r_diff["p_hln"] < 0.01 and r_same["p_hln"] > 0.9
    ok = ok and r_diff["mean_diff"] > 0
    return {
        "dm_stat": r_diff["dm"],
        "dm_hln": r_diff["dm_hln"],
        "p_alt": r_diff["p_hln"],
        "p_null": r_same["p_hln"],
        "score": float(ok),
    }
