"""Cross-impact regression of price changes on multi-asset order flow.

Extends the single-asset OFI impact regression to a system: for each asset
i, regress Δp_i on the signed flow of every asset j, yielding an impact
matrix Λ where Λ[i, j] measures how much asset j's flow moves asset i.

- ``cross_impact_matrix`` — stacked OLS with intercept per asset; returns
  the K×K impact matrix, per-equation R², and residuals;
- ``diagonal_dominance`` — the fraction of variance in Δp explained by own
  flow relative to cross flow: (Λ_ii − mean_j≠i |Λ_ij|) / max|Λ|;
- ``cross_impact_bootstrap_p`` — block-bootstrap two-sided p-value for
  H0: cross effects (off-diagonal block) are jointly zero.

Honesty: impact matrices estimated on synthetic coupled tapes are
correctness fixtures, not market evidence.

References:
- Cont, R., Cucuringu, M., Zhang, C. (2023). Cross-impact of order flow
  imbalance — the regression framework and state dependence.
- Hasbrouck, J. (1991). Measuring the information content of stock trades
  — the vector-autoregressive spirit of cross-effects.

Composition: pure numpy; deterministic seeds.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def cross_impact_matrix(
    price_changes: FloatArray,
    flows: FloatArray,
) -> dict[str, FloatArray | float]:
    """Regress each asset's price change on all assets' signed flows.

    ``price_changes`` and ``flows`` are both (T, K); returns the K×K impact
    matrix (row = response asset, column = flow asset), the per-equation R²
    vector, and the (T, K) residual matrix.
    """
    dp = np.asarray(price_changes, dtype=np.float64)
    x = np.asarray(flows, dtype=np.float64)
    if dp.ndim != 2 or x.shape != dp.shape:
        raise ValueError("price_changes and flows must be (T, K) of equal shape")
    t_total, k = dp.shape
    if t_total <= k + 2:
        raise ValueError("need more observations than assets")
    design = np.column_stack([np.ones(t_total), x])
    coef, _, _, _ = np.linalg.lstsq(design, dp, rcond=None)
    intercepts = coef[0]
    lam = coef[1:].T  # (K, K): row i = response of asset i to all flows
    fitted = design @ coef
    resid = dp - fitted
    ss_res = np.sum(resid * resid, axis=0)
    ss_tot = np.sum((dp - dp.mean(axis=0)) ** 2, axis=0)
    r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)
    return {
        "impact": np.asarray(lam, dtype=np.float64),
        "intercepts": np.asarray(intercepts, dtype=np.float64),
        "r2": np.asarray(r2, dtype=np.float64),
        "residuals": np.asarray(resid, dtype=np.float64),
    }


def diagonal_dominance(impact: FloatArray) -> float:
    """(mean own impact − mean |cross impact|) / max|impact| in [−1, 1]-ish."""
    lam = np.asarray(impact, dtype=np.float64)
    if lam.ndim != 2 or lam.shape[0] != lam.shape[1]:
        raise ValueError("impact must be a square matrix")
    k = lam.shape[0]
    if k < 2:
        raise ValueError("need at least two assets")
    own = np.abs(np.diag(lam)).mean()
    off = np.abs(lam - np.diag(np.diag(lam))).sum() / (k * (k - 1))
    scale = float(np.max(np.abs(lam)))
    if scale <= 0:
        return 0.0
    return float((own - off) / scale)


def cross_impact_bootstrap_p(
    price_changes: FloatArray,
    flows: FloatArray,
    *,
    block: int = 10,
    n_boot: int = 300,
    seed: int = 0,
) -> dict[str, float]:
    """Block-bootstrap p-value for the joint null of zero cross-impact.

    Statistic: Σ_i Σ_{j≠i} Λ_ij² (off-diagonal energy). The bootstrap
    resamples time blocks of the *flow* matrix independently per asset,
    breaking cross-asset alignment while preserving each asset's marginal
    serial dependence.
    """
    dp = np.asarray(price_changes, dtype=np.float64)
    x = np.asarray(flows, dtype=np.float64)
    if dp.ndim != 2 or x.shape != dp.shape:
        raise ValueError("price_changes and flows must be (T, K) of equal shape")
    t_total, k = dp.shape
    if t_total < 2 * block:
        raise ValueError("series too short for the requested block")
    obs = cross_impact_matrix(dp, x)["impact"]
    assert isinstance(obs, np.ndarray)
    stat_obs = float(np.sum(obs**2) - np.sum(np.diag(obs) ** 2))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        xb = np.empty_like(x)
        for j in range(k):
            starts = rng.integers(0, t_total, size=int(np.ceil(t_total / block)))
            idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:t_total] % t_total
            xb[:, j] = x[idx, j]
        lam_b = cross_impact_matrix(dp, xb)["impact"]
        assert isinstance(lam_b, np.ndarray)
        boots[b] = float(np.sum(lam_b**2) - np.sum(np.diag(lam_b) ** 2))
    # Null-percentile p-value: the bootstrap breaks cross-asset alignment,
    # so it samples the no-cross-impact world directly.
    p = float(np.mean(boots >= stat_obs))
    return {"stat": stat_obs, "p": p}
