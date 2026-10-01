"""Kyle's lambda: price-impact coefficient on the ZI-LOB trade stream.

Kyle (1985): prices respond linearly to net signed order flow,
``dP_t = lambda * q_t + eps_t`` where ``q_t`` is the signed aggressor
volume. On a unit-lot ZI-LOB every trade is one lot, so the signed
volume of a window is simply the net count of buys minus sells.

Per-trade price changes are dominated by bid-ask bounce, so the flow
is bucketed over fixed event-count windows and the regression runs on
``(net_signed_qty, trade-price change)`` per bucket — the standard
Kyle/Hasbrouck (1991) aggregation. Standard errors are Newey-West HAC
because consecutive window returns overlap through the book's mean
reversion.

The bench runs identical sessions under two Markov flow regimes (calm
vs trend-persistent) and reports per-regime lambda with HAC confidence
intervals. Evidence class: SYNTHETIC — the simulator's lambda is a
property of its own mechanics, not of any real venue.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
    RegimeState,
    ZILobConfig,
    ZILobSimulator,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

KYLE_LAMBDA_SCHEMA = "kyle_lambda.v1"


def _signed_qty(trades: list[Any]) -> NDArray[np.float64]:
    return np.asarray([1.0 if tr.aggressor == "buy" else -1.0 for tr in trades])


def bucket_flow(trades: list[Any], window: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Aggregate the trade stream into event-count windows.

    Returns (net_signed_qty, price_change) per completed window.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    signs = _signed_qty(trades)
    prices = np.asarray([tr.price for tr in trades], dtype=float)
    n = len(signs)
    if n < 2 * window + 1:
        raise ValueError(f"need >= {2 * window + 1} trades, got {n}")
    q_list: list[float] = []
    dp_list: list[float] = []
    for start in range(0, n - window, window):
        end = start + window
        q_list.append(float(np.sum(signs[start:end])))
        dp_list.append(float(prices[end] - prices[start]))
    return np.asarray(q_list), np.asarray(dp_list)


@dataclass(frozen=True)
class KyleFit:
    lambda_hat: float  # price units per unit signed lot
    se_nw: float  # Newey-West HAC standard error
    t_stat: float
    r2: float
    n_windows: int
    mean_abs_q: float


def _nw_se(x: NDArray[np.float64], resid: NDArray[np.float64], lag: int) -> float:
    """HAC variance of the OLS slope under a Bartlett kernel."""
    n = len(x)
    sxx = float(np.sum(x * x))
    scores = x * resid
    gamma0 = float(np.sum(scores * scores))
    meat = gamma0
    for j in range(1, lag + 1):
        w = 1.0 - j / (lag + 1)
        gamma_j = float(np.sum(scores[j:] * scores[: n - j]))
        meat += 2.0 * w * gamma_j
    var_beta = meat / (sxx * sxx)
    return math.sqrt(max(var_beta, 0.0))


def kyle_regression(
    q: NDArray[np.float64], dp: NDArray[np.float64], *, nw_lag: int | None = None
) -> KyleFit:
    """OLS ``dp = lambda * q + eps`` (no intercept — Kyle's model) with NW SE."""
    n = len(q)
    if n < 10:
        raise ValueError("need >= 10 windows")
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(dp))):
        raise ValueError("non-finite flow/returns")
    if float(np.sum(q * q)) == 0.0:
        raise ValueError("degenerate signed flow (all windows net zero)")
    lam = float(np.sum(q * dp) / np.sum(q * q))
    resid = dp - lam * q
    lag = nw_lag if nw_lag is not None else max(1, int(n ** (1 / 3)))
    se = _nw_se(q, resid, lag)
    t_stat = lam / se if se > 0 else float("inf") * (1.0 if lam > 0 else -1.0)
    ss_tot = float(np.sum(dp * dp))
    r2 = 1.0 - float(np.sum(resid * resid)) / ss_tot if ss_tot > 0 else float("nan")
    return KyleFit(
        lambda_hat=lam,
        se_nw=se,
        t_stat=t_stat,
        r2=r2,
        n_windows=n,
        mean_abs_q=float(np.mean(np.abs(q))),
    )


def lambda_session(
    *,
    n_mo: int,
    flow: MarkovRegimeFlow,
    bucket: int = 20,
    config: ZILobConfig | None = None,
    seed: int = 0,
) -> KyleFit:
    """Run a ZI-LOB session under ``flow`` and fit Kyle's lambda on its trades."""
    cfg = config or ZILobConfig(seed=seed)
    sim = ZILobSimulator(cfg, flow=flow)
    for _ in range(n_mo):
        sim.step()
    q, dp = bucket_flow(sim.trades, bucket)
    return kyle_regression(q, dp)


def kyle_lambda_bench(
    *,
    n_mo: int = 4000,
    bucket: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """Lambda under calm vs trend-persistent flow regimes.

    The two regimes share the book mechanics; they differ only in
    arrival intensity and buy-bias persistence. The bench reports both
    estimates with HAC CIs and whether the intervals overlap — a coarse
    cross-regime contrast, not a causal claim.
    """
    regimes = {
        "calm": MarkovRegimeFlow(
            states=[
                RegimeState(name="quiet", intensity_mult=0.8, p_buy=0.5),
                RegimeState(name="busy", intensity_mult=1.4, p_buy=0.5),
            ],
            stay_probs=[0.9, 0.9],
            seed=seed,
        ),
        "trend": MarkovRegimeFlow(
            states=[
                RegimeState(name="calm", intensity_mult=1.0, p_buy=0.5),
                RegimeState(name="trend_buy", intensity_mult=1.6, p_buy=0.8),
            ],
            stay_probs=[0.9, 0.85],
            seed=seed,
        ),
    }
    fits: dict[str, Any] = {}
    for name, flow in regimes.items():
        fit = lambda_session(n_mo=n_mo, flow=flow, bucket=bucket, seed=seed)
        lo = fit.lambda_hat - 1.96 * fit.se_nw
        hi = fit.lambda_hat + 1.96 * fit.se_nw
        fits[name] = {
            "lambda_hat": fit.lambda_hat,
            "se_nw": fit.se_nw,
            "t_stat": fit.t_stat,
            "r2": fit.r2,
            "n_windows": fit.n_windows,
            "mean_abs_q": fit.mean_abs_q,
            "ci95": [lo, hi],
        }
    calm_ci = fits["calm"]["ci95"]
    trend_ci = fits["trend"]["ci95"]
    ci_overlap = not (trend_ci[1] < calm_ci[0] or calm_ci[1] < trend_ci[0])
    payload = {
        "schema": KYLE_LAMBDA_SCHEMA,
        "kind": "kyle_lambda",
        "n_mo_per_regime": n_mo,
        "bucket_events": bucket,
        "seed": seed,
        "regimes": fits,
        "ci_overlap": ci_overlap,
        "positive_in_both": fits["calm"]["lambda_hat"] > 0 and fits["trend"]["lambda_hat"] > 0,
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "SYNTHETIC"
    payload["research_only"] = True
    payload["payload_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
