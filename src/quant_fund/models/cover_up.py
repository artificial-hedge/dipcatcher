"""Cover's universal portfolio.

References
----------
- Cover, T.M. (1991). "Universal Portfolios." *Mathematical Finance*
  1(1), 1-29.
- Cover, T.M. & Ordentlich, E. (1996). "Universal Portfolios with Side
  Information." *IEEE Transactions on Information Theory* 42(2), 348-363.
- Ordentlich, E. & Cover, T.M. (1998). "The Cost of Achieving the Best
  Portfolio in Hindsight." *Mathematics of Operations Research* 23(4),
  960-982.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence. The universal portfolio guarantees only
near-maximal *log-wealth* relative to the best constant-rebalanced
portfolio in hindsight — a relative, regret-style bound; it is not a
return prediction.

Composition notes
-----------------
The universal portfolio holds weights

    b_{t+1} = int b * S_t(b) dmu(b) / int S_t(b) dmu(b),
    S_t(b) = prod_{s<=t} (1 + b . r_s)

over the simplex under a Dirichlet(alpha) prior mu. Its wealth is the
mixture of all CRP wealths and tracks the hindsight-optimal CRP to
within O(sqrt(log T)) worst-case regret — no stationarity or
distributional assumptions on returns. The simplex integral is evaluated
on a deterministic barycentric grid (denominator ``grid`` per simplex
dimension); weights update by multiplicative accumulation of the
weighted wealth. The bench feeds a drifting-vs-flat two-asset tape where
the hindsight CRP concentrates on the winner.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _simplex_grid(k: int, denom: int) -> FloatArray:
    """Barycentric grid on the (k-1)-simplex with denominator ``denom``."""
    out: list[list[int]] = []
    cur = np.zeros(k, dtype=np.int64)

    def rec(rem: int, i: int) -> None:
        if i == k - 1:
            cur[i] = rem
            out.append(cur.tolist())
            return
        for v in range(rem + 1):
            cur[i] = v
            rec(rem - v, i + 1)

    rec(denom, 0)
    grid_pts = (np.asarray(out, dtype=np.float64) + 0.5) / float(denom)
    return grid_pts / grid_pts.sum(axis=1, keepdims=True)


def universal_portfolio(
    returns: FloatArray,
    alpha: float = 0.5,
    grid: int = 20,
) -> dict[str, float]:
    """Track the best CRP in hindsight over a (t, k) relative-return tape.

    ``returns`` holds *simple gross relatives* (price relatives, e.g.
    1.01 for +1%). Wealth products stay finite for any tape with
    strictly positive relatives.
    """
    rr = np.asarray(returns, dtype=np.float64)
    if rr.ndim != 2 or rr.shape[0] < 10 or rr.shape[1] < 2:
        raise ValueError("returns must be (t>=10, k>=2)")
    if not np.all(np.isfinite(rr)) or np.any(rr <= 0.0):
        raise ValueError("returns must be finite positive relatives")
    if not 0 < alpha < 5:
        raise ValueError("alpha out of range")
    if grid < 2:
        raise ValueError("grid too small")

    t, k = rr.shape
    pts = _simplex_grid(k, grid)
    n_pts = pts.shape[0]
    if n_pts > 20000:
        raise ValueError("simplex grid too dense")
    prior = np.prod(pts ** (alpha - 1.0), axis=1)
    prior = prior / prior.sum()

    s = np.ones(n_pts)
    b_path = np.zeros((t + 1, k))
    b_path[0] = np.ones(k) / k
    wealth = np.ones(t + 1)
    for i in range(t):
        s = s * (pts @ rr[i])
        tot = float(prior @ s)
        if tot <= 0:
            raise ValueError("degenerate wealth")
        wealth[i + 1] = tot
        b_path[i + 1] = (prior * s) @ pts / tot

    best_s = float(np.max(s))
    return {
        "final_wealth": float(wealth[-1]),
        "best_crp_wealth": best_s,
        "log_regret": float(np.log(best_s) - np.log(wealth[-1])),
        "w_max": float(np.max(b_path[-1])),
        "n_grid": float(n_pts),
    }


def synth_portfolio(
    t: int = 300,
    seed: int = 20261231 + 278,
    drift: float = 0.008,
) -> dict[str, FloatArray]:
    """Two-asset tape: asset 2 drifts upward, asset 1 is flat noise.

    Hindsight CRP concentrates on asset 2; the universal portfolio's
    terminal weight on it should dominate its start-of-day half.
    """
    rng = np.random.default_rng(seed)
    if t < 10:
        raise ValueError("t too small")
    r1 = 1.0 + rng.normal(0.0, 0.02, t)
    r2 = 1.0 + drift + rng.normal(0.0, 0.02, t)
    return {"returns": np.column_stack([np.clip(r1, 0.5, 1.5), np.clip(r2, 0.5, 1.5)])}


def bench_cover_up(seed: int = 20261231 + 278) -> dict[str, float]:
    """Wave-48 self-check: universal portfolio approaches the hindsight-
    optimal CRP on a drifting tape."""
    d = synth_portfolio(seed=seed)
    a = universal_portfolio(np.asarray(d["returns"]))
    b = universal_portfolio(np.asarray(d["returns"]))
    detects = float(a["w_max"] > 0.6 and a["log_regret"] < 1.0)
    return {
        "synthetic_detects": detects,
        "synthetic_determinism": float(a == b),
        "synthetic_final_wealth": a["final_wealth"],
        "synthetic_best_crp": a["best_crp_wealth"],
        "synthetic_log_regret": a["log_regret"],
        "synthetic_w_max": a["w_max"],
        "synthetic_n_grid": a["n_grid"],
    }
