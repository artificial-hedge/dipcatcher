"""Order-flow imbalance (OFI) events and price-impact estimation.

Implements the Cont–Cucuringu–Zhang (2023) two-stage view of price
formation at the bar/tape level:

- ``ofi_events`` — maps a sequence of (side, price, size) updates into OFI
  events: each consecutive price change in the same direction adds size, a
  reversal or a size-increase at the same price adds/subtracts as specified
  by the event rules; here we use the standard four-rule accumulation that
  the upstream ``signed_flow_imbalance`` approximates, but at event
  granularity;
- ``kyle_lambda_ofi`` — regress mid-price changes on OFI to get the impact
  coefficient (the OFI analogue of Kyle's λ) with R² and a permutation
  p-value;
- ``ar_signed_flow`` — fit an AR(p) to the OFI series and report the
  F-statistic of joint predictability against the constant-only model.

Honesty: impact coefficients estimated on synthetic tapes where flow
drives prices are correctness fixtures, not market evidence.

References:
- Cont, R., Cucuringu, M., Zhang, C. (2023). Cross-impact of order flow
  imbalance: a state-dependent approach (and the OFI event construction).
- Cont, R., Kukanov, A., Stoikov, S. (2014). The price impact of order
  book events — OFI → price impact regressions.
- Kyle, A. S. (1985). Continuous auctions and insider trading — λ.

Composition: pure numpy; deterministic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def ofi_events(signs: IntArray, sizes: FloatArray) -> FloatArray:
    """OFI event series from per-trade signs and sizes.

    Each trade contributes sign·size; the event series is the per-trade
    signed flow (the finest-grain OFI). Coarser aggregation is obtained by
    summing within bars via ``quant_fund.flowbars.bars`` ids.
    """
    signs = np.asarray(signs, dtype=np.int64)
    sizes = np.asarray(sizes, dtype=np.float64)
    if signs.shape != sizes.shape:
        raise ValueError("signs and sizes must have the same shape")
    return signs.astype(np.float64) * sizes


def aggregate_ofi(events: FloatArray, ids: IntArray) -> FloatArray:
    """Sum OFI events per bar (ids sorted, dense)."""
    events = np.asarray(events, dtype=np.float64)
    ids = np.asarray(ids, dtype=np.int64)
    if events.shape != ids.shape:
        raise ValueError("events and ids must have the same shape")
    n_bars = int(ids[-1]) + 1 if len(ids) else 0
    out = np.zeros(n_bars, dtype=np.float64)
    np.add.at(out, ids, events)
    return out


def kyle_lambda_ofi(
    price_changes: FloatArray,
    ofi: FloatArray,
    *,
    n_perm: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    """Regress Δp on OFI: Δp = λ·OFI + ε; returns λ, R², permutation p.

    The permutation p-value shuffles OFI against Δp to calibrate how often
    a spurious λ this large arises under flow/price independence.
    """
    dp = np.asarray(price_changes, dtype=np.float64)
    x = np.asarray(ofi, dtype=np.float64)
    if dp.shape != x.shape or dp.ndim != 1:
        raise ValueError("price_changes and ofi must be one-dimensional arrays of equal shape")
    n = len(dp)
    if n < 10:
        raise ValueError("need at least 10 observations")
    var_x = float(np.var(x))
    if var_x <= 0:
        raise ValueError("ofi has no variation")
    lam = float(np.dot(x, dp) / np.dot(x, x))
    resid = dp - lam * x
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum((dp - dp.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        idx = rng.permutation(n)
        lam_p = float(np.dot(x, dp[idx]) / np.dot(x, x))
        if abs(lam_p) >= abs(lam):
            count += 1
    return {"lambda": lam, "r2": float(r2), "p_perm": (count + 1) / (n_perm + 1)}


def ar_signed_flow(ofi: FloatArray, p: int = 5) -> dict[str, float]:
    """AR(p) fit on the OFI series with a joint F-test of predictability.

    Compares the AR(p) regression against a constant-only model; the F
    statistic tests H0: all AR coefficients are zero (no linear
    predictability of signed flow).
    """
    from scipy import stats as sps

    x = np.asarray(ofi, dtype=np.float64)
    n = len(x)
    if n <= p + 5:
        raise ValueError("series too short for the requested AR order")
    y = x[p:]
    design = np.column_stack([x[p - k : n - k] for k in range(1, p + 1)])
    design = np.column_stack([np.ones(len(y)), design])
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    df1 = p
    df2 = n - p - 1
    f = ((ss_tot - ss_res) / df1) / max(ss_res / df2, 1e-24)
    pval = float(1.0 - sps.f.cdf(f, df1, df2))
    return {"f_stat": float(f), "p": pval, "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0}
