"""Black-Karasinski short-rate lattice — Black & Karasinski (1991).

The BK model evolves the log of the short rate:

    d ln r = (theta(t) - a ln r) dt + sigma dW

so rates stay positive and theta(t) is calibrated to the initial
zero curve. The lattice is a recombining trinomial tree in
x = ln r - mean drift, with node spacing dx = sigma sqrt(3 dt).
From node j the expected next position is m = j(1 - a dt); the
tree branches to (k-1, k, k+1) with k = round(m) and probabilities
matched to the increment's first two moments:

    s = m - k,  pm = 2/3 - s^2,  pu = (1/3 + s^2 + s)/2,
    pd = (1/3 + s^2 - s)/2

Arrow-Debreu prices Q_{i,j} (value of a claim paying 1 at node
(i,j)) are swept forward; the level displacement alpha_i is solved
so that sum_j Q_{i,j} exp(-exp(alpha_i + j dx) dt) = P(0, t_{i+1})
— the input discount curve reprices exactly by construction.

References
----------
- Black, F., Karasinski, P. (1991). "Bond and option pricing when
  short rates are lognormal." *Financial Analysts Journal* 47(4).
- Hull, J.C. *Options, Futures and Other Derivatives* ch. 31 —
  trinomial moment matching used here.
- Clewlow, L., Strickland, C. (1998). *Implementing Derivative
  Models* — Arrow-Debreu / Jamshidian forward induction.

Honesty
-------
SYNTHETIC curve and caplet only; bench verifies exact bond repricing
and caplet vol monotonicity — not a market claim.

Composition
-----------
Called by ``quant_fund.research.benches_w66.bench_black_karasinski``.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

FloatArray = NDArray[np.float64]


def _trinomial_probs(j: float, a: float, dt: float) -> tuple[float, float, float, int]:
    """From node position j (in dx units) return (pu, pm, pd, k)."""
    m = j * (1.0 - a * dt)
    k = int(round(m))
    s = m - k
    pm = 2.0 / 3.0 - s * s
    pu = 0.5 * (1.0 / 3.0 + s * s + s)
    pd = 0.5 * (1.0 / 3.0 + s * s - s)
    return pu, pm, pd, k


def bk_tree(
    a: float,
    sigma: float,
    curve_times: FloatArray,
    df: FloatArray,
    dt: float = 0.05,
) -> dict[str, object]:
    """Calibrated BK trinomial lattice.

    Returns dict with ``dt``, ``dx``, ``alpha`` (per-level
    displacement), ``ad`` (list of Arrow-Debreu vectors), ``jmax``
    (half-width per level), ``rates`` (list of node-rate vectors).
    """
    curve_times = np.asarray(curve_times, dtype=float)
    df = np.asarray(df, dtype=float)
    if not (a > 0 and sigma > 0 and dt > 0):
        raise ValueError("a, sigma, dt must be positive")
    if curve_times.ndim != 1 or curve_times.size < 2 or np.any(df <= 0):
        raise ValueError("bad discount curve")
    t_end = float(curve_times[-1])
    n_steps = max(int(round(t_end / dt)), 1)
    dt = t_end / n_steps

    def log_df(tt: float) -> float:
        if tt <= curve_times[0]:
            return float(np.log(df[0]) * tt / curve_times[0])
        return float(np.interp(tt, curve_times, np.log(df)))

    dx = sigma * np.sqrt(3.0 * dt)
    ad: list[FloatArray] = [np.array([1.0])]  # Q_{0,0} = 1
    alphas: list[float] = []
    rates: list[FloatArray] = []
    # Node index array for level i: j in [-i..i]
    for i in range(n_steps):
        j = np.arange(-i, i + 1, dtype=float)
        q = ad[i]
        target = np.exp(log_df((i + 1) * dt))

        def p_of_alpha(alpha: float, j: FloatArray = j, q: FloatArray = q) -> float:
            r = np.exp(alpha + j * dx)
            return float(np.sum(q * np.exp(-r * dt)))

        alpha0 = np.log(-log_df(dt) / dt) if i == 0 else alphas[-1]
        lo, hi = alpha0 - 6.0, alpha0 + 6.0
        if p_of_alpha(lo) < target or p_of_alpha(hi) > target:
            raise ValueError("alpha bracket failed")
        alpha = brentq(
            lambda aa, target=target: p_of_alpha(aa) - target,
            lo,
            hi,
            xtol=1e-13,
        )
        alphas.append(alpha)
        r_i = np.exp(alpha + j * dx)
        rates.append(r_i)
        # Propagate Q to level i+1 (positions -(i+1)..(i+1)).
        q_next = np.zeros(2 * i + 3)
        for jj in range(j.size):
            pu, pm, pd, k = _trinomial_probs(float(j[jj]), a, dt)
            w = q[jj] * np.exp(-r_i[jj] * dt)
            # node index in next-level array: pos k in [-i-1..i+1]
            base = i + 1 + k
            q_next[base + 1] += w * pu
            q_next[base] += w * pm
            q_next[base - 1] += w * pd
        ad.append(q_next)

    return {
        "dt": np.array([dt]),
        "dx": np.array([dx]),
        "alpha": np.array(alphas),
        "ad": ad,
        "rates": rates,
    }


def bk_bond(tree: dict[str, object], maturity_steps: int) -> float:
    """P(0, t_m) implied by the lattice — equals input curve."""
    ad = cast(list[FloatArray], tree["ad"])
    if maturity_steps >= len(ad):
        raise ValueError("maturity beyond lattice")
    return float(np.sum(ad[maturity_steps]))


def bk_caplet(
    a: float,
    sigma: float,
    curve_times: FloatArray,
    df: FloatArray,
    expiry: float,
    strike: float,
    dt: float = 0.05,
) -> float:
    """Caplet on the dt short rate fixing at ``expiry`` (paid arrears)."""
    if not (expiry > 0 and strike >= 0):
        raise ValueError("bad caplet spec")
    tree = bk_tree(a, sigma, curve_times, df, dt)
    dt_arr = cast(FloatArray, tree["dt"])
    dt_v = float(dt_arr[0])
    step = int(round(expiry / dt_v))
    ad = cast(list[FloatArray], tree["ad"])
    rates = cast(list[FloatArray], tree["rates"])
    q = np.asarray(ad[step], dtype=np.float64)
    r = np.asarray(rates[step], dtype=np.float64)
    payoff = np.maximum(r - strike, 0.0) * dt_v
    # paid one period later: discount each node's own short rate.
    return float(np.sum(q * np.exp(-r * dt_v) * payoff))


def bench_black_karasinski(seed: int = 20261231 + 385) -> dict[str, float]:
    """SYNTHETIC check — lattice reprices the curve; caplet sane."""
    _ = np.random.default_rng(seed)
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r_flat = 0.04
    df = np.exp(-r_flat * times * (1.0 + 0.1 * np.sin(times)))
    a, sigma = 0.15, 0.10
    tree = bk_tree(a, sigma, times, df, dt=0.25)

    def log_df(tt: float) -> float:
        if tt <= times[0]:
            return float(np.log(df[0]) * tt / times[0])
        return float(np.interp(tt, times, np.log(df)))

    errs = []
    for m in range(1, 20):
        p = bk_bond(tree, m)
        tgt = float(np.exp(log_df(m * float(cast(FloatArray, tree["dt"])[0]))))
        errs.append(abs(p - tgt))
    bond_err = float(np.max(errs))
    if bond_err > 1e-8:
        raise ValueError("BK bond repricing failed")
    c1 = bk_caplet(a, sigma, times, df, 2.0, 0.04, dt=0.25)
    c2 = bk_caplet(a, 2.0 * sigma, times, df, 2.0, 0.04, dt=0.25)
    if not (c1 > 0 and c2 > c1):
        raise ValueError("caplet vol monotonicity failed")
    rmin = min(float(np.min(r)) for r in cast(list[FloatArray], tree["rates"]))
    if rmin <= 0:
        raise ValueError("non-positive rates on lattice")
    return {
        "synthetic_bk_bond_err": bond_err,
        "synthetic_bk_caplet": c1,
        "synthetic_bk_caplet_vol_ratio": c2 / c1,
        "synthetic_bk_min_rate": rmin,
        "synthetic_score": 1.0,
    }
