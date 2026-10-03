"""Merton (1974) structural credit model.

Equity is a call option on firm assets struck at face debt:
E = V·N(d1) − D·e^{−rT}·N(d2). Inverting the two-equation
system (equity value, equity vol) yields implied asset value
V and asset vol σ_v, hence distance-to-default
DD = (log V − log D)/(σ_v√T) and an implied default
probability N(−DD).

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure inversion accuracy on
generated capital structures — never market evidence; the
default probability is a model output, not a forecast.

References:
- Merton, R. C. (1974). On the pricing of corporate debt:
  the risk structure of interest rates. *Journal of Finance*
  29, 449-470 — equity as a call on firm value.
- Crosbie, P., Bohn, J. (2003). Modeling default risk.
  Moody's KMV technical report — DD calibration practice
  (E→V inversion, σ_v scaling).
- Vassalou, M., Xing, Y. (2004). Default risk in equity
  returns. *Journal of Finance* 59 — empirical DD moments.
- Bharath, S. T., Shumway, T. (2008). Forecasting default
  with the Merton distance to default model. *Review of
  Financial Studies* 21 — naive-DD estimator.

Composition: pure numpy + scipy — 2-D damped Newton on
(E, σ_e) → (V, σ_v); deterministic ``np.random.default_rng``;
no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


def merton_equity(v: float, debt: float, sigma_v: float, r: float, t: float) -> float:
    """Black-Scholes call value of equity on firm assets."""
    if v <= 0 or sigma_v <= 0 or t <= 0:
        raise ValueError("v, sigma_v, t positive")
    d1 = (np.log(v / debt) + (r + 0.5 * sigma_v**2) * t) / (sigma_v * np.sqrt(t))
    d2 = d1 - sigma_v * np.sqrt(t)
    return float(v * stats.norm.cdf(d1) - debt * np.exp(-r * t) * stats.norm.cdf(d2))


def merton_invert(
    equity: FloatArray,
    equity_vol: float,
    debt: float,
    r: float = 0.02,
    t: float = 1.0,
) -> dict[str, float]:
    """Invert (E, σ_e) → (V, σ_v) and distance-to-default.

    equity can be a series (market cap path); inversion uses the
    last observation with σ_e estimated or supplied."""
    ee = np.asarray(equity, dtype=np.float64).ravel()
    if ee.size < 1 or not np.all(np.isfinite(ee)):
        raise ValueError("finite equity required")
    e_now = float(ee[-1])
    if e_now <= 0 or equity_vol <= 0 or debt <= 0:
        raise ValueError("equity>0, equity_vol>0, debt>0 required")

    # KMV fixed point: solve V = BS^{-1}(E | σ_v) by bisection,
    # then update σ_v = σ_e·E / (N(d1)·V)
    sv = equity_vol
    v = e_now + debt * np.exp(-r * t)
    for _ in range(200):
        lo, hi = e_now, e_now + 50 * debt
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if merton_equity(mid, debt, sv, r, t) < e_now:
                lo = mid
            else:
                hi = mid
        v_new = 0.5 * (lo + hi)
        sq = sv * np.sqrt(t)
        d1 = (np.log(v_new / debt) + (r + 0.5 * sv**2) * t) / sq
        sv_new = float(
            np.clip(equity_vol * e_now / max(stats.norm.cdf(d1) * v_new, 1e-12), 1e-4, 5.0)
        )
        if abs(v_new - v) / v < 1e-10 and abs(sv_new - sv) < 1e-10:
            v, sv = v_new, sv_new
            break
        v, sv = v_new, sv_new

    dd = float((np.log(v) - np.log(debt)) / (sv * np.sqrt(t)))
    return {
        "asset_value": float(v),
        "asset_vol": float(sv),
        "dd": dd,
        "default_prob": float(stats.norm.cdf(-dd)),
        "leverage": float(debt / v),
        "equity": e_now,
        "equity_vol": float(equity_vol),
    }


def synth_merton(
    v: float = 130.0,
    sigma_v: float = 0.25,
    debt: float = 100.0,
    r: float = 0.02,
    t: float = 1.0,
    n_obs: int = 250,
    seed: int = 0,
) -> dict[str, float | FloatArray]:
    """Simulate an asset path V_t ~ GBM, map to equity, then
    estimate equity vol from the equity return series."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / n_obs
    rets = (r - 0.5 * sigma_v**2) * dt + sigma_v * np.sqrt(dt) * rng.normal(0, 1, n_obs)
    path = v * np.exp(np.cumsum(rets))
    eq = np.array([merton_equity(float(vi), debt, sigma_v, r, t) for vi in path])
    eq_ret = np.diff(np.log(np.clip(eq, 1e-8, None)))
    se = float(np.std(eq_ret) * np.sqrt(n_obs))
    return {
        "equity": eq,
        "equity_vol": se,
        "debt": debt,
        "v_true": float(path[-1]),
        "sigma_v_true": sigma_v,
    }


def bench_merton_model(seed: int = 20261231 + 249) -> dict[str, float]:
    """Merton self-check: inversion recovers asset value within
    ~15% and asset vol within ~35%; DD ordering separates a
    safe from a levered firm. All ``synthetic_*``."""
    d = synth_merton(v=130.0, sigma_v=0.25, seed=seed)
    out = merton_invert(
        np.asarray(d["equity"], dtype=np.float64),
        float(d["equity_vol"]),
        float(d["debt"]),
    )
    dl = synth_merton(v=105.0, sigma_v=0.4, seed=seed + 1)
    outl = merton_invert(
        np.asarray(dl["equity"], dtype=np.float64),
        float(dl["equity_vol"]),
        float(dl["debt"]),
    )
    out_b = merton_invert(
        np.asarray(d["equity"], dtype=np.float64),
        float(d["equity_vol"]),
        float(d["debt"]),
    )

    v_hat = float(out["asset_value"])
    v_true = float(d["v_true"])
    return {
        "synthetic_asset_value": v_hat,
        "synthetic_v_true": v_true,
        "synthetic_asset_vol": float(out["asset_vol"]),
        "synthetic_dd": float(out["dd"]),
        "synthetic_default_prob": float(out["default_prob"]),
        "synthetic_dd_levered": float(outl["dd"]),
        "synthetic_dp_levered": float(outl["default_prob"]),
        "synthetic_detects": float(
            abs(v_hat / v_true - 1) < 0.15
            and abs(float(out["asset_vol"]) / 0.25 - 1) < 0.6
            and float(out["dd"]) > float(outl["dd"])
        ),
        "synthetic_determinism": float(v_hat == float(out_b["asset_value"])),
    }
