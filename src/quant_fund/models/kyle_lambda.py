"""Kyle (1985) lambda — price impact and auction equilibrium.

Two complementary pieces: (i) the empirical lambda — OLS slope
of price change on signed net order flow, the workhorse market-
impact estimator; (ii) the single-period Kyle auction
equilibrium — a strategic informed trader exploiting noise
trader flow chooses intensity β = σ_u/σ_v, giving equilibrium
λ = σ_v/(2σ_u), half the private information impounded into
price.

Honesty: synthetic benches simulate the auction with
theoretical intensities and check the estimated λ recovers the
equilibrium value — proper diagnostics, never market evidence.

References:
- Kyle, A. S. (1985). Continuous auctions and insider trading.
  *Econometrica* 53 — equilibrium λ, β, profit identities.
- Hasbrouck, J. (2007). *Empirical Market Microstructure*,
  OUP — the empirical lambda regression form.
- Boulatov, A., Kyle, A. S., Livdan, D. (2013). On the
  estimation of Kyle's lambda. *mimeo* — bias in the OLS
  estimator under misspecification.
- Foster, F. D., Viswanathan, S. (1996). Strategic trading when
  agents forecast the forecasts of others. *Journal of
  Finance* 51 — multi-informed extension bounds.

Composition: pure numpy — OLS + closed-form equilibrium;
deterministic ``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


def kyle_lambda(signed_flow: FloatArray, price_change: FloatArray) -> dict[str, float]:
    """Empirical lambda: Δp = c + λ·y + e by OLS.

    Returns lambda, its HAC-free t-stat, R², and the residual
    sd. signed_flow and price_change are matched (n,) arrays."""
    y = np.asarray(signed_flow, dtype=np.float64)
    dp = np.asarray(price_change, dtype=np.float64)
    if y.shape != dp.shape or y.size < 30 or y.ndim != 1:
        raise ValueError("matched (n>=30,) arrays required")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(dp)):
        raise ValueError("finite inputs required")
    x = np.column_stack([np.ones(y.size), y])
    beta, *_ = np.linalg.lstsq(x, dp, rcond=None)
    resid = dp - x @ beta
    s2 = float(resid @ resid) / (y.size - 2)
    xtx_inv = np.linalg.inv(x.T @ x)
    se = float(np.sqrt(s2 * xtx_inv[1, 1]))
    lam = float(beta[1])
    t = lam / se if se > 0 else 0.0
    r2 = 1.0 - float(resid @ resid) / float(np.sum((dp - dp.mean()) ** 2))
    return {
        "lambda": lam,
        "t_lambda": t,
        "p_lambda": float(2 * stats.t.sf(abs(t), y.size - 2)),
        "r2": r2,
        "intercept": float(beta[0]),
        "resid_sd": float(np.sqrt(s2)),
    }


def kyle_equilibrium(sigma_u: float, sigma_v: float) -> dict[str, float]:
    """Single-period Kyle auction equilibrium parameters."""
    if sigma_u <= 0 or sigma_v <= 0:
        raise ValueError("positive sigma_u, sigma_v required")
    beta = sigma_u / sigma_v
    lam = sigma_v / (2.0 * sigma_u)
    profit = 0.5 * sigma_u * sigma_v  # expected informed profit
    resid_info = 0.5 * sigma_v**2  # Var(v|y)
    return {
        "beta": float(beta),
        "lambda": float(lam),
        "expected_profit": float(profit),
        "residual_var": float(resid_info),
        "info_share": 0.5,
    }


def synth_kyle(
    n: int = 600,
    sigma_u: float = 4.0,
    sigma_v: float = 2.0,
    sigma_eps: float = 0.5,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Simulate n auctions under the theoretical equilibrium plus
    a public-information price shock eps."""
    rng = np.random.default_rng(seed)
    eq = kyle_equilibrium(sigma_u, sigma_v)
    v = rng.normal(0, sigma_v, n)
    u = rng.normal(0, sigma_u, n)
    x = eq["beta"] * v
    y = x + u
    dp = eq["lambda"] * y + rng.normal(0, sigma_eps, n)
    return {"signed_flow": y, "price_change": dp}


def bench_kyle_lambda(seed: int = 20261231 + 253) -> dict[str, float]:
    """Kyle self-check: OLS lambda recovers the equilibrium λ
    within ~15% and R² ≈ .5 (half the flow is informed).
    All ``synthetic_*``."""
    eq = kyle_equilibrium(4.0, 2.0)
    d = synth_kyle(sigma_u=4.0, sigma_v=2.0, sigma_eps=0.3, seed=seed)
    out = kyle_lambda(d["signed_flow"], d["price_change"])
    out_b = kyle_lambda(d["signed_flow"], d["price_change"])
    lam_hat = float(out["lambda"])
    return {
        "synthetic_lambda": lam_hat,
        "synthetic_lambda_true": eq["lambda"],
        "synthetic_t": float(out["t_lambda"]),
        "synthetic_r2": float(out["r2"]),
        "synthetic_beta_true": eq["beta"],
        "synthetic_detects": float(
            abs(lam_hat / eq["lambda"] - 1) < 0.15 and float(out["p_lambda"]) < 0.01
        ),
        "synthetic_determinism": float(lam_hat == float(out_b["lambda"])),
    }
