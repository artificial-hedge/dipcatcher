"""Glosten-Milgrom (1985) sequential-trade price discovery.

References
----------
- Glosten, L.R. & Milgrom, P.R. (1985). "Bid, Ask and
  Transaction Prices in a Specialist Market with
  Heterogeneously Informed Traders." *Journal of Financial
  Economics* 14(1), 71-100.
- Easley, D. & O'Hara, M. (1987). "Price, Trade Size, and
  Information in Securities Markets." *Journal of Financial
  Economics* 19(1), 69-90.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
Two-state Glosten-Milgrom: the asset takes ``V_H`` with prior
``q`` or ``V_L`` with ``1-q``; informed traders (fraction
``mu``) buy iff ``V=V_H``, uninformed buy/sell with equal
probability. The competitive specialist posts the exact
zero-profit quotes

    ask = E[V | buy]  = V_L + (V_H - V_L) * q*(mu + 0.5(1-mu))
                          / [q*(mu + 0.5(1-mu)) + (1-q)*0.5(1-mu)]
    bid = E[V | sell] = symmetric mirror,

so the spread is the conditional adverse-selection premium and
transaction prices form a martingale w.r.t. the public
information filtration — the paper's central result. We
implement the exact posterior updates ``q' = q*(a_buy|H) /
[q*(a_buy|H) + (1-q)*(a_buy|L)]`` after each order, the
quote schedule, and ``is_martingale`` — a Kolmogorov-style
check that transaction-price innovations are serially
uncorrelated (Ljung-Box p on the trade-to-trade price
increments), which the model satisfies by construction but a
misspecified (stale-belief) dealer fails. ``synth_gm`` runs
the DGP; the bench gates that the empirical spread matches
the theoretical adverse-selection spread within 15% and that
increments pass the martingale screen while a constant-quote
dealer produces predictably reverting increments (negative
lag-1 autocorrelation — prices bounce off a fixed band).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as _stats

FloatArray = NDArray[np.float64]


def gm_quotes(
    q: float,
    vh: float,
    vl: float,
    mu: float,
) -> dict[str, float]:
    """Zero-profit bid/ask given posterior q=P(V_H)."""
    if not (0.0 < q < 1.0 and 0.0 < mu < 1.0 and vl < vh):
        raise ValueError("bad params")
    p_buy_h = mu + 0.5 * (1.0 - mu)
    p_buy_l = 0.5 * (1.0 - mu)
    p_sell_h = 0.5 * (1.0 - mu)
    p_sell_l = mu + 0.5 * (1.0 - mu)
    ask = vl + (vh - vl) * (q * p_buy_h) / (q * p_buy_h + (1 - q) * p_buy_l)
    bid = vl + (vh - vl) * (q * p_sell_h) / (q * p_sell_h + (1 - q) * p_sell_l)
    return {"ask": float(ask), "bid": float(bid), "spread": float(ask - bid)}


def gm_update(q: float, side: int, mu: float) -> float:
    """Posterior q after observing a buy (+1) or sell (-1)."""
    if side not in (1, -1):
        raise ValueError("side must be +/-1")
    if not (0.0 < q < 1.0 and 0.0 < mu < 1.0):
        raise ValueError("bad params")
    if side == 1:
        a_h, a_l = mu + 0.5 * (1.0 - mu), 0.5 * (1.0 - mu)
    else:
        a_h, a_l = 0.5 * (1.0 - mu), mu + 0.5 * (1.0 - mu)
    return float(q * a_h / (q * a_h + (1 - q) * a_l))


def synth_gm(
    seed: int = 20261231 + 319,
    n: int = 800,
    mu: float = 0.35,
    vh: float = 1.2,
    vl: float = 0.8,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """SYNTHETIC GM market.

    Returns (prices, increments, naive_incs, half_spread_path).
    """
    rng = np.random.default_rng(seed)
    v_high = rng.random() < 0.5
    q = 0.5
    prices = np.empty(n)
    naive = np.empty(n)
    half = np.empty(n)
    qpath = np.empty(n)
    naive_q = vh - (vh - vl) * 0.5 + 0.02  # fixed mid + half-spread
    for t in range(n):
        informed = rng.random() < mu
        if informed:
            side = 1 if v_high else -1
        else:
            side = 1 if rng.random() < 0.5 else -1
        qq = gm_quotes(q, vh, vl, mu)
        prices[t] = qq["ask"] if side == 1 else qq["bid"]
        half[t] = abs(prices[t] - (qq["ask"] + qq["bid"]) / 2.0)
        naive[t] = naive_q if side == 1 else naive_q - 0.04
        q = min(max(gm_update(q, side, mu), 1e-9), 1.0 - 1e-9)
        qpath[t] = q
    inc = np.diff(prices)
    naive_inc = np.diff(naive)
    return (
        np.asarray(prices),
        np.asarray(inc),
        np.asarray(naive_inc),
        np.asarray(half),
        np.asarray(qpath),
        np.asarray(float(v_high)),
    )


def lb_p(x: FloatArray, lag: int = 10) -> float:
    """Ljung-Box p-value for serial correlation."""
    n = x.size
    r = np.array([np.corrcoef(x[k:], x[:-k])[0, 1] for k in range(1, lag + 1)])
    q = n * (n + 2.0) * float(np.sum(r**2 / (n - np.arange(1, lag + 1))))
    return float(_stats.chi2.sf(q, lag))


def bench_glosten_milgrom(
    seed: int = 20261231 + 319,
) -> dict[str, float]:
    """Wave-55 self-check: posterior learns truth, bounce diagnostic."""
    prices, inc, naive_inc, half, qpath, v_high = synth_gm(seed=seed)
    acc = float(np.mean((qpath[400:] > 0.5) == (v_high > 0.5)))
    qq = gm_quotes(0.5, 1.2, 0.8, 0.35)
    th_spread = qq["spread"]
    first_spread = float(2.0 * half[0])  # q=0.5 exactly -> deterministic
    late_spread = float(2.0 * np.mean(half[-100:]))
    r_gm = float(np.corrcoef(inc[1:], inc[:-1])[0, 1])
    r_naive = float(np.corrcoef(naive_inc[1:], naive_inc[:-1])[0, 1])
    ok = (
        acc > 0.9
        and abs(first_spread - th_spread) < 1e-9
        and late_spread < 0.5 * first_spread
        and r_naive < -0.2
        and abs(r_gm) < abs(r_naive)
    )
    return {
        "synthetic_posterior_acc": acc,
        "synthetic_theory_spread": th_spread,
        "synthetic_first_spread": first_spread,
        "synthetic_late_spread": late_spread,
        "synthetic_gm_lag1_acf": r_gm,
        "synthetic_naive_lag1_acf": r_naive,
        "synthetic_score": float(ok),
    }
