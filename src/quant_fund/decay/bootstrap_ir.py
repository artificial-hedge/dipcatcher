"""Bootstrap inference for the information ratio of an IC series.

The IR (mean IC / std IC) is a ratio statistic with a skewed finite-sample
distribution; a plain t-interval on the mean IC ignores ratio uncertainty.
This module bootstraps the whole IR:

- ``bootstrap_ir`` — circular block bootstrap of the IC series → percentile
  CI for IR, its standard error, and P(IR ≤ 0) (the "skill" exceedance
  probability);
- ``bootstrap_ir_delta`` — normal-approx interval via the delta method on
  (mean, std) as a cheap cross-check.

Honesty: intervals are bootstrap approximations on the supplied series; the
stationary/block bootstrap preserves serial correlation up to the block.

References:
- Politis, D. N., Romano, J. P. (1994). The stationary bootstrap.
- Efron, B., Tibshirani, R. J. (1993). *An Introduction to the Bootstrap* —
  percentile intervals and the delta method.

Composition: numpy + scipy (locked); deterministic seeds.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def information_ratio(ic: FloatArray) -> float:
    """IR = mean(IC) / std(IC); NaN when the series has no variation."""
    x = np.asarray(ic, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        raise ValueError("need at least 3 valid IC observations")
    sd = float(np.std(x, ddof=1))
    if sd <= 0:
        return float("nan")
    return float(np.mean(x) / sd)


def bootstrap_ir(
    ic: FloatArray,
    *,
    block: int = 5,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Block-bootstrap distribution of the IR.

    Returns the point estimate, bootstrap standard error, the central 90%
    percentile interval, and P(IR ≤ 0) estimated from the bootstrap draws.
    """
    x = np.asarray(ic, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2 * block:
        raise ValueError("series too short for the requested block")
    obs = information_ratio(x)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        xb = x[idx]
        sd = float(np.std(xb, ddof=1))
        draws[b] = float(np.mean(xb) / sd) if sd > 0 else np.nan
    draws = draws[np.isfinite(draws)]
    if len(draws) < n_boot // 2:
        raise ValueError("bootstrap produced too few finite IR draws")
    lo, hi = np.percentile(draws, [5.0, 95.0])
    return {
        "ir": obs,
        "se": float(np.std(draws, ddof=1)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_nonpositive": float(np.mean(draws <= 0.0)),
    }


def bootstrap_ir_delta(ic: FloatArray) -> dict[str, float]:
    """Delta-method normal interval for the IR.

    With μ̂ = mean, σ̂ = std, the gradient of IR = μ/σ is (1/σ, −μ/(2σ²)) on
    (μ, σ²)-style parameters; the variance uses the plug-in moment estimates.
    """
    x = np.asarray(ic, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        raise ValueError("need at least 10 valid IC observations")
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    if sd <= 0:
        return {
            "ir": float("nan"),
            "se": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
        }
    ir = mu / sd
    var_mu = sd * sd / n
    m2 = float(np.mean((x - mu) ** 2))
    m4 = float(np.mean((x - mu) ** 4))
    var_s2 = (m4 - m2 * m2) / n
    cov_mu_s2 = float(np.mean((x - mu) * ((x - mu) ** 2 - m2))) / n
    g_mu = 1.0 / sd
    g_s2 = -mu / (2.0 * sd**3)
    var_ir = g_mu * g_mu * var_mu + g_s2 * g_s2 * var_s2 + 2.0 * g_mu * g_s2 * cov_mu_s2
    se = float(np.sqrt(max(var_ir, 0.0)))
    from scipy import stats as sps

    z = sps.norm.ppf(0.95)
    return {
        "ir": float(ir),
        "se": se,
        "ci_lo": float(ir - z * se),
        "ci_hi": float(ir + z * se),
    }
