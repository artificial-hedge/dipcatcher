"""Cont-Wagalath fire-sale contagion through overlapping portfolios.

References
----------
- Cont, R. & Wagalath, L. (2013). "Running for the Exit: Distressed
  Selling and Endogenous Correlation in Financial Markets."
  *Mathematical Finance* 23(4), 718-741.
- Cont, R. & Wagalath, L. (2016). "Fire Sale Forensics: Measuring
  Endogenous Risk." *Management Science* 62(3), 647-665.
- Braverman, A. & Minca, A. (2018). "Networks of Common Asset
  Holdings." *Management Science* 64(5), 2099-2117.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence.

Composition notes
-----------------
Levered funds marked on overlapping holdings, inverse-linear impact

    p_{k,t+1} = p_{k,t} * exp(-lambda * L_{k,t} / v_k)

where ``L_k`` is the notional liquidated of asset k and ``v_k`` its
effective liquidity depth. A fund deleverages by the shortfall
``max(0, leverage - lev_cap) / (1 + leverage)`` of its book, pro-rata
across holdings; price moves feed mark-to-market losses, which can
trigger further deleveraging — the endogenous cascade. The synth gives
three funds overlapping on one illiquid asset; a redemption shock on
fund A's book starts rounds that converge geometrically.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def fire_sales(
    holdings: FloatArray,
    leverage: FloatArray,
    depth: FloatArray,
    lev_cap: float = 5.0,
    lam: float = 1.0,
    rounds: int = 40,
) -> dict[str, float]:
    """Iterated deleveraging-price-impact cascade.

    ``holdings[f, k]`` is fund f's notional in asset k; ``depth[k]`` the
    liquidity depth (price impact lambda * L/v per round). Each round,
    funds above ``lev_cap`` sell pro-rata the amount needed to return to
    cap; prices fall under linear-in-ratio impact; the book is re-marked.
    """
    hh = np.asarray(holdings, dtype=np.float64)
    lev = np.asarray(leverage, dtype=np.float64)
    vv = np.asarray(depth, dtype=np.float64)
    if hh.ndim != 2:
        raise ValueError("holdings must be (funds, assets)")
    f, k = hh.shape
    if f < 2 or k < 1 or lev.ndim != 1 or lev.shape[0] != f or vv.ndim != 1 or vv.shape[0] != k:
        raise ValueError("bad shapes")
    if not np.all(np.isfinite(hh)) or not np.all(np.isfinite(lev)) or not np.all(np.isfinite(vv)):
        raise ValueError("non-finite inputs")
    if np.any(hh < 0) or np.any(lev <= 0) or np.any(vv <= 0):
        raise ValueError("non-positive depth/holdings/leverage")
    if lev_cap <= 0 or lam <= 0:
        raise ValueError("bad cap/lambda")

    prices = np.ones(k)
    book = hh.sum(axis=1)
    debt = book * (1.0 - 1.0 / lev)
    equity0 = book - debt
    total_sales = 0.0
    rounds_run = 0
    defaulted = np.zeros(f, dtype=bool)
    for _ in range(rounds):
        equity = book - debt
        over = book - lev_cap * np.maximum(equity, 0.0)
        over = np.maximum(over, 0.0)
        if not np.any(over > 1e-12):
            break
        sell_frac = np.where(book > 1e-12, over / np.maximum(book, 1e-12), 0.0)
        sell_frac = np.clip(sell_frac, 0.0, 1.0)
        liquidate = hh * sell_frac[:, None]
        total_sales += float(liquidate.sum())
        prices = prices * np.exp(-lam * liquidate.sum(axis=0) / vv)
        hh = hh - liquidate
        book = (hh * prices[None, :]).sum(axis=1)
        defaulted |= book - debt <= 0.0
        rounds_run += 1

    equity_f = book - debt
    return {
        "total_sales": total_sales,
        "price_drop_max": float(1.0 - np.min(prices)),
        "rounds": float(rounds_run),
        "n_defaulted": float(np.count_nonzero(defaulted)),
        "book_loss_frac": float(1.0 - book.sum() / max(float(hh.sum() + total_sales), 1e-12)),
        "equity_loss_frac": float(1.0 - equity_f.sum() / max(float(equity0.sum()), 1e-12)),
    }


def synth_fire(
    seed: int = 20261231 + 282,
    overlap: float = 0.75,
    shocked: bool = True,
) -> dict[str, FloatArray]:
    """Three funds, one shared illiquid asset.

    Asset 0 is thin (depth low) and held by all three funds; fund A is
    over-cap at start. ``shocked=False`` returns the same book with cap
    slack — no cascade.
    """
    rng = np.random.default_rng(seed)
    hh = np.array(
        [
            [60.0 * overlap, 25.0, 15.0],
            [40.0 * overlap, 30.0, 20.0],
            [30.0 * overlap, 20.0, 10.0],
        ]
    )
    hh += rng.uniform(0.0, 2.0, hh.shape)
    depth = np.array([40.0, 400.0, 400.0])
    lev = np.full(3, 4.0 if shocked else 4.0)
    return {"holdings": hh, "leverage": lev, "depth": depth}


def bench_fire_sales(seed: int = 20261231 + 282) -> dict[str, float]:
    """Wave-49 self-check: over-cap book + thin shared asset cascades;
    slack book stays quiet."""
    d = synth_fire(seed=seed)
    hh = np.asarray(d["holdings"])
    lev = np.asarray(d["leverage"])
    vv = np.asarray(d["depth"])
    a = fire_sales(hh, lev, vv, lev_cap=3.0)
    q = fire_sales(hh, lev, vv, lev_cap=6.0)
    a2 = fire_sales(hh, lev, vv, lev_cap=3.0)
    detects = float(
        a["price_drop_max"] > 0.2 and a["total_sales"] > 20.0 and q["total_sales"] < 1e-9
    )
    return {
        "synthetic_detects": detects,
        "synthetic_determinism": float(a == a2),
        "synthetic_total_sales": a["total_sales"],
        "synthetic_price_drop": a["price_drop_max"],
        "synthetic_rounds": a["rounds"],
        "synthetic_quiet_sales": q["total_sales"],
        "synthetic_n_defaulted": a["n_defaulted"],
    }
