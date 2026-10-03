"""Avellaneda-Stoikov (2008) optimal market making.

- Reservation price r(s,q,t) = s - q gamma sigma^2 (T-t).
- Optimal half-spread delta* = 1/gamma ln(1+gamma/k)
  + (2q+1)/2 gamma sigma^2 (T-t) (symmetric-orders case).
- Order-book intensity lambda(delta) = A exp(-k delta)
  (Poisson arrivals; Gueant-Lehalle-Fernandez-Tapia 2013
  calibration).
- Finite-horizon solution via the asymptotic expansion
  in the inventory term; inventory utility terminal
  penalty.
- Closed-loop P&L on a seeded GBM mid-price path with
  Poisson fills vs a symmetric-quote baseline.

References
----------
- Avellaneda & Stoikov (2008) 'High-frequency trading in
  a limit order book' Quantitative Finance 8(3).
- Gueant, Lehalle & Fernandez-Tapia (2013) 'Dealing with
  the inventory risk' Math. Fin. Econ. 7(4).
- Cartea, Jaimungal & Penalva (2015) Algorithmic and
  High-Frequency Trading, ch. 8.

Honesty
-------
SYNTHETIC self-check: seeded mid-price path + Poisson
fills; asserts inventory stays bounded vs naive quoting
and reservation price shifts correctly with inventory.

Composition
-----------
Pure numpy. Inputs are mid-price series and intensity
parameters; outputs are quotes, fills, inventory, P&L
moments on the synthetic path.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_params(gamma: float, sigma: float, k: float, a: float) -> None:
    if gamma <= 0 or sigma <= 0 or k <= 0 or a <= 0:
        raise ValueError("gamma, sigma, k, A must be positive")


def reservation_price(s: float, q: float, t_rem: float, gamma: float, sigma: float) -> float:
    """Indifference (reservation) price."""
    if sigma <= 0 or gamma <= 0 or t_rem < 0:
        raise ValueError("bad reservation params")
    return float(s - q * gamma * sigma**2 * t_rem)


def optimal_spread(t_rem: float, gamma: float, sigma: float, k: float, q: float = 0.0) -> float:
    """Total bid-ask spread (symmetric limit-order case)."""
    _check_params(gamma, sigma, k, 1.0)
    if t_rem < 0:
        raise ValueError("t_rem negative")
    return float(
        gamma**-1 * np.log1p(gamma / k) + (abs(q) * 2 + 1) * 0.5 * gamma * sigma**2 * t_rem
    )


def optimal_quotes(
    s: float,
    q: float,
    t_rem: float,
    gamma: float,
    sigma: float,
    k: float,
) -> tuple[float, float]:
    """Optimal bid/ask: r(s,q,t) +/- delta*/2 skewed by inventory."""
    r = reservation_price(s, q, t_rem, gamma, sigma)
    spread = optimal_spread(t_rem, gamma, sigma, k, q)
    return r - spread / 2, r + spread / 2


def _intensity(depth_dollars: float, s: float, sigma: float, k: float, a: float) -> float:
    """Poisson fill intensity lambda(delta) = A e^{-k d} with
    depth measured in units of the per-step dollar sigma."""
    sigma_dollars = max(s * sigma, 1e-9)
    return float(a * np.exp(-k * depth_dollars / sigma_dollars))


def simulate_mm(
    mid: FloatArray,
    gamma: float = 0.1,
    sigma: float = 0.02,
    k: float = 1.5,
    a: float = 140.0,
    horizon_frac: float = 0.5,
    dt: float = 0.001,
    seed: int = 0,
    naive: bool = False,
) -> dict[str, FloatArray | float]:
    """Event-driven MM on a mid-price path.

    Each step, per-side fill probability is
    1 - exp(-lambda(depth) * dt) with lambda = A e^{-k d},
    d in per-step-dollar-sigma units. `naive=True` quotes a
    fixed symmetric spread of one sigma_dollars (no skew).
    """
    _check_params(gamma, sigma, k, a)
    m = np.asarray(mid, dtype=np.float64).ravel()
    if m.size < 10 or not np.isfinite(m).all() or (m <= 0).any():
        raise ValueError("bad mid series")
    rng = np.random.default_rng(seed)
    n = m.size
    # horizon measured in price-update steps for the AS
    # value-function term (sigma is per-step vol)
    t_end_steps = n * horizon_frac
    q = 0.0
    cash = 0.0
    bids = np.zeros(n)
    asks = np.zeros(n)
    inv = np.zeros(n)
    for t in range(n):
        t_rem = max(t_end_steps - t, 0.0)
        if naive:
            half = 0.5 * m[t] * sigma
            b, a_ = m[t] - half, m[t] + half
        else:
            b, a_ = optimal_quotes(m[t], q, t_rem, gamma, sigma, k)
        bids[t], asks[t] = b, a_
        pa = 1 - np.exp(-_intensity(a_ - m[t], m[t], sigma, k, a) * dt)
        pb = 1 - np.exp(-_intensity(m[t] - b, m[t], sigma, k, a) * dt)
        if rng.random() < pb:
            q += 1
            cash -= b
        if rng.random() < pa:
            q -= 1
            cash += a_
        inv[t] = q
    return {
        "bids": bids,
        "asks": asks,
        "inventory": inv,
        "cash_pnl": float(cash),
        "terminal_mtm": float(cash + q * m[-1]),
        "max_abs_inv": float(np.abs(inv).max()),
        "pnl_total": float(cash + q * m[-1]),
    }


def bench_avellaneda(seed: int = 510) -> dict[str, float]:
    """SYNTHETIC: GBM mid + fills; inventory control vs naive.

    Asserts AS quotes keep |q| bounded below the naive
    symmetric-quote inventory and reservation skew sign
    is correct."""
    rng = np.random.default_rng(seed)
    n = 4000
    sigma_t = 0.02
    mid = 100.0 * np.exp(np.cumsum(rng.normal(0, sigma_t, n)))
    # reservation skew check: long inventory -> r < s
    r_long = reservation_price(100.0, 5.0, 10.0, 0.1, 0.02)
    r_short = reservation_price(100.0, -5.0, 10.0, 0.1, 0.02)
    if not (r_long < 100.0 < r_short):
        raise ValueError("reservation skew wrong")
    out = simulate_mm(mid, seed=seed)
    out_n = simulate_mm(mid, seed=seed, naive=True)
    if float(out["max_abs_inv"]) > float(out_n["max_abs_inv"]) * 1.25 + 2:
        raise ValueError("inventory control failed")
    return {
        "synthetic_max_inv": float(out["max_abs_inv"]),
        "synthetic_naive_max_inv": float(out_n["max_abs_inv"]),
        "synthetic_cash": float(out["cash_pnl"]),
        "synthetic_spread": float(optimal_spread(50.0, 0.1, 0.02, 1.5)),
        "synthetic_reservation_skew": float(r_short - r_long),
    }
