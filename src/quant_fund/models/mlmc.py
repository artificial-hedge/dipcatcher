"""Multilevel Monte Carlo for SDE payoffs — Giles (2008).

The MLMC estimator telescopes the finest-level payoff:

    E[P_L] = E[P_0] + sum_{l=1..L} E[P_l - P_{l-1}]

with each difference level estimated on its own optimal sample
count N_l chosen to equalize marginal variance-reduction cost. For
Euler-Maruyama with step h_l = h_0 M^{-l} and a Lipschitz payoff,
Var[P_l - P_{l-1}] = O(h_l) so total cost to reach RMSE epsilon is
O(epsilon^{-2} (log epsilon)^2) vs O(epsilon^{-3}) plain MC.

This module simulates a scalar SDE (drift/diffusion callables),
applies an arbitrary payoff to the terminal value, and returns the
MLMC estimate with per-level diagnostics.

References
----------
- Giles, M.B. (2008). "Multilevel Monte Carlo path simulation."
  *Operations Research* 56(3), 607-617.
- Giles, M.B. (2015). "Multilevel Monte Carlo methods." *Acta
  Numerica* 24, 259-328 — review and complexity theorem.
- Giles, M.B., Szpruch, L. (2014). "Antithetic multilevel Monte Carlo
  estimation for multi-dimensional SDEs." *Annals of Applied Prob.*

Honesty
-------
SYNTHETIC only: the bench checks the estimator against the analytic
Black-76 price and the O(h) variance-decay rate — not a live claim.

Composition
-----------
Called by ``quant_fund.research.benches_w66.bench_mlmc``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
DriftFn = Callable[[FloatArray, float], FloatArray]
DiffFn = Callable[[FloatArray, float], FloatArray]
PayoffFn = Callable[[FloatArray], FloatArray]


def _euler_level(
    x0: float,
    t: float,
    n_steps: int,
    n_paths: int,
    mu: DriftFn,
    sigma: DiffFn,
    rng: np.random.Generator,
) -> FloatArray:
    dt = t / n_steps
    x = np.full(n_paths, x0)
    for i in range(n_steps):
        ti = i * dt
        z = rng.standard_normal(n_paths)
        x = x + mu(x, ti) * dt + sigma(x, ti) * np.sqrt(dt) * z
    return x


def _level_sample(
    level: int,
    n_paths: int,
    x0: float,
    t: float,
    steps0: int,
    factor: int,
    mu: DriftFn,
    sigma: DiffFn,
    payoff: PayoffFn,
    rng: np.random.Generator,
) -> FloatArray:
    """One iid draw of P_l - P_{l-1} using coupled coarse/fine sims."""
    if level == 0:
        return payoff(_euler_level(x0, t, steps0, n_paths, mu, sigma, rng))
    nf = steps0 * factor**level
    nc = steps0 * factor ** (level - 1)
    dt_f = t / nf
    dt_c = t / nc
    sub = nf // nc
    xf = np.full(n_paths, x0)
    xc = np.full(n_paths, x0)
    for i in range(nc):
        z_sum = np.zeros(n_paths)
        for j in range(sub):
            ti = i * dt_c + j * dt_f
            z = rng.standard_normal(n_paths)
            xf = xf + mu(xf, ti) * dt_f + sigma(xf, ti) * np.sqrt(dt_f) * z
            z_sum += z
        # Coarse step reuses summed innovations (variance = sub*dt_f = dt_c).
        xc = xc + mu(xc, i * dt_c) * dt_c + sigma(xc, i * dt_c) * np.sqrt(dt_f) * z_sum
    return payoff(xf) - payoff(xc)


def mlmc_estimate(
    x0: float,
    t: float,
    mu: DriftFn,
    sigma: DiffFn,
    payoff: PayoffFn,
    n_levels: int = 4,
    n_per_level: int = 20000,
    steps0: int = 8,
    factor: int = 4,
    seed: int = 0,
) -> dict[str, FloatArray | float]:
    """Fixed-N_l MLMC estimate with per-level means/variances."""
    if not (x0 > 0 and t > 0 and n_levels >= 1 and n_per_level >= 8 and steps0 >= 1):
        raise ValueError("bad mlmc configuration")
    rng = np.random.default_rng(seed)
    means = np.zeros(n_levels)
    variances = np.zeros(n_levels)
    for lev in range(n_levels):
        d = _level_sample(lev, n_per_level, x0, t, steps0, factor, mu, sigma, payoff, rng)
        means[lev] = float(np.mean(d))
        variances[lev] = float(np.var(d, ddof=1))
    estimate = float(np.sum(means))
    se = float(np.sqrt(np.sum(variances / n_per_level)))
    return {
        "estimate": estimate,
        "se": se,
        "level_means": means,
        "level_vars": variances,
    }


def variance_decay_rate(level_vars: FloatArray, steps0: int, factor: int) -> float:
    """Estimate beta in Var[P_l - P_{l-1}] ~ h_l^beta via log-log fit."""
    v = np.asarray(level_vars, dtype=float)
    if v.ndim != 1 or v.size < 3 or np.any(v <= 0):
        raise ValueError("need >=3 positive level variances")
    lv = np.arange(1, v.size)
    h = steps0 * factor**lv
    slope, _ = np.polyfit(np.log(h), np.log(v[1:]), 1)
    return float(-slope)


def bench_mlmc(seed: int = 20261231 + 384) -> dict[str, float]:
    """SYNTHETIC check — MLMC matches Black-76; variance decays ~O(h)."""
    s0, k, t, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    def mu(x: FloatArray, _t: float) -> FloatArray:
        return r * x

    def sig(x: FloatArray, _t: float) -> FloatArray:
        return sigma * x

    disc = np.exp(-r * t)

    def payoff(x: FloatArray) -> FloatArray:
        return np.asarray(disc * np.maximum(x - k, 0.0), dtype=np.float64)

    out = mlmc_estimate(s0, t, mu, sig, payoff, n_levels=4, n_per_level=20000, seed=seed)
    est = float(out["estimate"])
    se = float(out["se"])
    lv = np.asarray(out["level_vars"], dtype=float)
    from scipy.stats import norm

    sd = sigma * np.sqrt(t)
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma * sigma) * t) / sd
    d2 = d1 - sd
    bs = s0 * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
    err = abs(est - bs)
    if err > max(4.0 * se, 0.15):
        raise ValueError("MLMC estimate outside MC noise band")
    beta = variance_decay_rate(lv, 8, 4)
    if beta < 0.5:  # Lipschitz payoff + Euler -> ~1.0; allow slack
        raise ValueError("variance decay rate too low")
    return {
        "synthetic_mlmc_est": est,
        "synthetic_mlmc_se": se,
        "synthetic_mlmc_bs_err": err,
        "synthetic_mlmc_var_decay": beta,
        "score": 1.0,
    }
