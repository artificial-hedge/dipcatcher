"""IC significance with serial dependence and trial multiplicity.

Three corrections on top of the plain IC t-statistic:

- block bootstrap p-value: resamples the IC series in circular blocks to
  respect serial correlation;
- deflated comparison: when n_trials correlated signals were tried, the
  relevant null is not "IC mean = 0" but "max IC mean over trials = 0".
  ``expected_max_null_tstat`` computes E[max of n standard-normal t-stats]
  by exact quadrature (e.g. E[max of 2] = 1/√π), and ``deflated_excess_t``
  reports by how much the observed t exceeds that expectation.

Honesty: these are null-model corrections for multiple/dependent testing;
they do not rescue a signal that fails out of sample.

References:
- Bailey, D. H., López de Prado, M. (2014). The deflated Sharpe ratio:
  correcting for selection bias, backtest overfitting and non-normality —
  the max-of-trials null this module ports to IC.
- Harvey, C. R., Liu, Y. (2015). Backtesting — multiple testing and
  serial-correlation adjustments.
- Politis, D. N., Romano, J. P. (1994). The stationary bootstrap — circular
  block resampling.

Composition: numpy + scipy.stats (locked); deterministic seeds.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import integrate
from scipy import stats as sps

FloatArray = NDArray[np.float64]


def expected_max_null_tstat(n_trials: int) -> float:
    """E[max of n i.i.d. standard-normal statistics], exact quadrature.

    E[max] = ∫ x · n Φ(x)^{n−1} φ(x) dx. Valid for any n ≥ 1 (n = 1 → 0;
    n = 2 → 1/√π exactly).
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return 0.0
    n = float(n_trials)

    def integrand(x: float) -> float:
        return float(x * n * sps.norm.cdf(x) ** (n - 1) * sps.norm.pdf(x))

    val, _ = integrate.quad(integrand, -np.inf, np.inf, epsabs=1e-10)
    return float(val)


def deflated_excess_t(t_obs: float, n_trials: int, ic_std_over_sqrt_t: float) -> dict[str, float]:
    """Observed t minus the expected max-t under the null, in IC units.

    ``ic_std_over_sqrt_t`` is the per-trial standard error of the IC mean
    (σ_IC / √T). Returns the excess in both t units and IC units plus a
    normal-approximation one-sided p-value (the distribution of the max
    itself is only approximately normal; documented as an approximation).
    """
    if ic_std_over_sqrt_t <= 0:
        raise ValueError("ic_std_over_sqrt_t must be positive")
    e_max = expected_max_null_tstat(n_trials)
    excess_t = float(t_obs - e_max)
    return {
        "expected_max_t": e_max,
        "excess_t": excess_t,
        "excess_ic": float(excess_t * ic_std_over_sqrt_t),
        "p_approx": float(1.0 - sps.norm.cdf(excess_t)),
    }


def block_bootstrap_ic_pvalue(
    ic: FloatArray,
    *,
    n_boot: int = 1000,
    block: int = 5,
    seed: int = 0,
) -> dict[str, float]:
    """Circular block-bootstrap two-sided p-value for the IC mean.

    The resampled distribution of the block-bootstrapped mean accounts for
    serial correlation up to the block length.
    """
    x = np.asarray(ic, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < block * 2:
        raise ValueError("series too short for the requested block length")
    if n_boot < 100:
        raise ValueError("n_boot must be >= 100 for a stable p-value")
    rng = np.random.default_rng(seed)
    obs = float(np.mean(x))
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        boots[b] = float(np.mean(x[idx]))
    centered = boots - float(np.mean(boots))
    denom = float(np.std(centered, ddof=1))
    if denom <= 0:
        return {"p": 0.0 if obs != 0 else 1.0, "boot_std": 0.0, "obs": obs}
    z = float(obs / denom)
    p = float(2.0 * (1.0 - sps.norm.cdf(abs(z))))
    return {"p": p, "boot_std": denom, "obs": obs}
